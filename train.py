import os
import time
import argparse

import paddle

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

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    tranforms = [
        SampleFrames(clip_len=16, frame_interval=1, num_clips=1),
        RawFrameDecode(),
        Resize(scale=(127, 171)),
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
        Resize(scale=(127, 171)),
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
    load_pretrained_model(model, '/Users/alex/Desktop/c3d.pdparams')


    batch_size = 30
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
                                      batch_size=batch_size // 2, shuffle=False, drop_last=False, return_list=True)
    lr = paddle.optimizer.lr.MultiStepDecay(learning_rate=1.25e-4, milestones=[20, 40], gamma=0.1)
    optimizer = paddle.optimizer.SGD(learning_rate=lr, weight_decay=5e-4, parameters=model.parameters())

    avg_loss = 0.0
    avg_acc = 0.0
    max_epochs = 45
    epoch = 0
    log_iters = 1
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()

    iters = iters_per_epoch * max_epochs
    iter = 0
    batch_start = time.time()
    best_accuracy = 0.0
    while epoch < max_epochs:
        model.train()
        for batch_id, data in enumerate(train_loader):
            reader_cost_averager.record(time.time() - batch_start)
            iter += 1

            outputs = model.train_step(data, optimizer)
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            model.clear_gradients()

            log_vars = outputs['log_vars']
            avg_loss += log_vars['loss']
            avg_acc += log_vars['top1_acc']

            batch_cost_averager.record(
                time.time() - batch_start, num_samples=batch_size)
            if iter % log_iters == 0:
                avg_loss /= log_iters
                avg_acc /= log_iters
                remain_iters = iters - iter
                avg_train_batch_cost = batch_cost_averager.get_average()
                avg_train_reader_cost = reader_cost_averager.get_average()
                eta = calculate_eta(remain_iters, avg_train_batch_cost)

                print(
                    "[TRAIN] epoch={}, batch_id={}, loss={:.6f}, lr={:.6f},acc={:.3f} ETA {}"
                        .format(epoch, batch_id + 1,
                                avg_loss, optimizer.get_lr(), avg_acc, eta))
                avg_loss = 0.0
                avg_acc = 0.0
                avg_pose_acc = 0.0
                reader_cost_averager.reset()
                batch_cost_averager.reset()
        lr.step()

        val_avg_loss = 0.0
        val_avg_acc = 0.0
        for batch_id, data in enumerate(val_loader):
            with paddle.no_grad():
                outputs = model.val_step(data, optimizer)
                log_vars = outputs['log_vars']
                val_avg_loss += log_vars['loss']
                val_avg_acc += log_vars['top1_acc']
                val_avg_loss /= log_iters
                val_avg_acc /= log_iters
                print("[EVAL] epoch={}, batch_id={}, loss={:.6f},acc={:.3f}".format(epoch, batch_id + 1, val_avg_loss,
                                                                                    val_avg_acc))
        if val_avg_acc > best_accuracy:
            best_accuracy = val_avg_acc
            current_save_dir = os.path.join("output", 'best_model')
            if not os.path.exists(current_save_dir):
                os.makedirs(current_save_dir)
            paddle.save(model.state_dict(),
                        os.path.join(current_save_dir, 'model.pdparams'))
            paddle.save(optimizer.state_dict(),
                        os.path.join(current_save_dir, 'model.pdopt'))
        epoch += 1
