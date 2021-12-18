import os
import time
import argparse
import random

import paddle
import numpy as np

from datasets import SampleFrames, RawFrameDecode, Resize, RandomCrop, CenterCrop, Flip, Normalize, FormatShape, Collect
from datasets import RawframeDataset
from timer import TimeAverager, calculate_eta
from models.c3d import C3D
from models.i3d_head import I3DHead
from models.recognizer3d import Recognizer3D
from utils import load_pretrained_model


def parse_args():
    parser = argparse.ArgumentParser(description='Model training')

    parser.add_argument(
        '--dataset_root',
        dest='dataset_root',
        help='The path of dataset root',
        type=str,
        default='/Users/alex/baidu/mmaction2/data/ucf101/')

    parser.add_argument(
        '--pretrained',
        dest='pretrained',
        help='The pretrained of model',
        type=str,
        default=None)

    parser.add_argument(
        '--resume',
        dest='resume',
        help='The path of resume model',
        type=str,
        default=None
    )

    parser.add_argument(
        '--last_epoch',
        dest='last_epoch',
        help='The last epoch of resume model',
        type=int,
        default=-1
    )

    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='batch_size',
        type=int,
        default=32
    )

    parser.add_argument(
        '--max_epochs',
        dest='max_epochs',
        help='max_epochs',
        type=int,
        default=100
    )

    parser.add_argument(
        '--log_iters',
        dest='log_iters',
        help='log_iters',
        type=int,
        default=10
    )

    parser.add_argument(
        '--seed',
        dest='seed',
        help='random seed',
        type=int,
        default=1234
    )


    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    paddle.seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    tranforms = [
        SampleFrames(clip_len=16, frame_interval=1, num_clips=1),
        RawFrameDecode(),
        Resize(scale=(128, 171)),
        RandomCrop(size=112),
        Flip(flip_ratio=0.5),
        Normalize(mean=[104, 117, 128], std=[1, 1, 1], to_bgr=False),
        FormatShape(input_format='NCTHW'),
        Collect(keys=['imgs', 'label'], meta_keys=[])
    ]
    dataset = RawframeDataset(ann_file=os.path.join(args.dataset_root, 'ucf101_train_split_1_rawframes.txt'),
                              pipeline=tranforms, data_prefix=os.path.join(args.dataset_root, "rawframes"))

    val_tranforms = [
        SampleFrames(clip_len=16, frame_interval=1, num_clips=1, test_mode=True),
        RawFrameDecode(),
        Resize(scale=(128, 171)),
        CenterCrop(crop_size=112),
        Normalize(mean=[104, 117, 128], std=[1, 1, 1], to_bgr=False),
        FormatShape(input_format='NCTHW'),
        Collect(keys=['imgs', 'label'], meta_keys=[])
    ]
    val_dataset = RawframeDataset(ann_file=os.path.join(args.dataset_root, 'ucf101_val_split_1_rawframes.txt'),
                                  pipeline=val_tranforms, data_prefix=os.path.join(args.dataset_root, "rawframes"),
                                  test_mode=True)

    backbone = C3D(dropout_ratio=0.5, init_std=0.005)
    head = I3DHead(num_classes=101, in_channels=4096, spatial_type=None, dropout_ratio=0.5, init_std=0.01)
    model = Recognizer3D(backbone=backbone, cls_head=head)
    if args.pretrained is not None:
        load_pretrained_model(model, args.pretrained)

    batch_size = args.batch_size
    train_loader = paddle.io.DataLoader(
        dataset,
        num_workers=0,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        return_list=True,
    )

    iters_per_epoch = len(train_loader)
    val_loader = paddle.io.DataLoader(val_dataset,
                                      batch_size=batch_size, shuffle=False, drop_last=False, return_list=True)

    max_epochs = args.max_epochs
    if args.last_epoch > -1:
        last_epoch = (args.last_epoch + 1) * iters_per_epoch
    else:
        last_epoch = args.last_epoch
    learning_rate = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=1e-3, T_max=max_epochs * iters_per_epoch, last_epoch=last_epoch)
    lr = paddle.optimizer.lr.LinearWarmup(
        learning_rate=learning_rate,
        warmup_steps=300,
        start_lr=0,
        end_lr=1e-3,
        last_epoch=last_epoch)
    grad_clip = paddle.nn.ClipGradByNorm(40)
    optimizer = paddle.optimizer.Momentum(learning_rate=lr, weight_decay=5e-4, parameters=model.parameters(), grad_clip=grad_clip)

    if args.resume is not None:
        if os.path.exists(args.resume):
            resume_model = os.path.normpath(args.resume)
            ckpt_path = os.path.join(resume_model, 'model.pdparams')
            para_state_dict = paddle.load(ckpt_path)
            ckpt_path = os.path.join(resume_model, 'model.pdopt')
            opti_state_dict = paddle.load(ckpt_path)
            model.set_state_dict(para_state_dict)
            optimizer.set_state_dict(opti_state_dict)

    epoch = 1

    log_iters = args.log_iters
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()

    iters = iters_per_epoch * max_epochs
    iter = 0
    batch_start = time.time()
    best_accuracy = -0.01
    while epoch <= max_epochs:
        total_loss = 0.0
        total_acc = 0.0
        model.train()
        for batch_id, data in enumerate(train_loader):
            reader_cost_averager.record(time.time() - batch_start)
            iter += 1

            outputs = model.train_step(data, optimizer)
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            model.clear_gradients()
            lr.step()
            log_vars = outputs['log_vars']
            total_loss += log_vars['loss']
            total_acc += log_vars['top1_acc']

            batch_cost_averager.record(
                time.time() - batch_start, num_samples=batch_size)
            if iter % log_iters == 0:
                avg_loss = total_loss / (batch_id + 1)
                avg_acc = total_acc / (batch_id + 1)
                remain_iters = iters - iter
                avg_train_batch_cost = batch_cost_averager.get_average()
                avg_train_reader_cost = reader_cost_averager.get_average()
                eta = calculate_eta(remain_iters, avg_train_batch_cost)

                print(
                    "[TRAIN] epoch={}, batch_id={}, loss={:.6f}, lr={:.6f},acc={:.3f},"
                    "avg_reader_cost: {} sec, avg_batch_cost: {} sec, avg_samples: {}, avg_ips: {} images/sec  ETA {}"
                        .format(epoch, batch_id + 1,
                                avg_loss, optimizer.get_lr(), avg_acc,
                                avg_train_reader_cost, avg_train_batch_cost,
                                batch_size, batch_size / avg_train_batch_cost,
                                eta))
                reader_cost_averager.reset()
                batch_cost_averager.reset()
            batch_start = time.time()

        model.eval()
        results = []
        total_val_avg_loss = 0.0
        total_val_avg_acc = 0.0
        for batch_id, data in enumerate(val_loader):
            with paddle.no_grad():
                # outputs = model.val_step(data, optimizer)
                imgs = data['imgs']
                label = data['label']
                result = model(imgs, label, return_loss=False)
            results.extend(result)
        print(f"[EVAL] epoch={epoch}")
        key_score = val_dataset.evaluate(results, metrics=['top_k_accuracy', 'mean_class_accuracy'])

        if key_score['top1_acc'] > best_accuracy:
            best_accuracy = key_score['top1_acc']
            current_save_dir = os.path.join("output", 'best_model')
            if not os.path.exists(current_save_dir):
                os.makedirs(current_save_dir)
            paddle.save(model.state_dict(),
                        os.path.join(current_save_dir, 'model.pdparams'))
            paddle.save(optimizer.state_dict(),
                        os.path.join(current_save_dir, 'model.pdopt'))
        epoch += 1
