import os.path as osp
import copy as cp
from collections.abc import Sequence
from collections import abc
import collections
from itertools import repeat

import cv2
from cv2 import (IMREAD_COLOR, IMREAD_GRAYSCALE, IMREAD_IGNORE_ORIENTATION,
                 IMREAD_UNCHANGED)
import numpy as np

from datasets.file_client import FileClient

imread_flags = {
    'color': IMREAD_COLOR,
    'grayscale': IMREAD_GRAYSCALE,
    'unchanged': IMREAD_UNCHANGED,
    'color_ignore_orientation': IMREAD_IGNORE_ORIENTATION | IMREAD_COLOR,
    'grayscale_ignore_orientation':
    IMREAD_IGNORE_ORIENTATION | IMREAD_GRAYSCALE
}

cv2_interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}


def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.

    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def is_tuple_of(seq, expected_type):
    """Check whether it is a tuple of some type.

    A partial method of :func:`is_seq_of`.
    """
    return is_seq_of(seq, expected_type, seq_type=tuple)


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)

def imfrombytes(content, flag='color', channel_order='bgr', backend=None):
    img_np = np.frombuffer(content, np.uint8)
    flag = imread_flags[flag] if isinstance(flag, str) else flag
    img = cv2.imdecode(img_np, flag)
    if flag == IMREAD_COLOR and channel_order == 'rgb':
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    return img

def imresize(img,
             size,
             return_scale=False,
             interpolation='bilinear',
             out=None,
             backend=None):
    """Resize image to a given size.

    Args:
        img (ndarray): The input image.
        size (tuple[int]): Target size (w, h).
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend.
        out (ndarray): The output destination.
        backend (str | None): The image resize backend type. Options are `cv2`,
            `pillow`, `None`. If backend is None, the global imread_backend
            specified by ``mmcv.use_backend()`` will be used. Default: None.

    Returns:
        tuple | ndarray: (`resized_img`, `w_scale`, `h_scale`) or
            `resized_img`.
    """
    h, w = img.shape[:2]

    resized_img = cv2.resize(
        img, size, dst=out, interpolation=cv2_interp_codes[interpolation])
    if not return_scale:
        return resized_img
    else:
        w_scale = size[0] / w
        h_scale = size[1] / h
        return resized_img, w_scale, h_scale


def _scale_size(size, scale):
    """Rescale a size by a ratio.

    Args:
        size (tuple[int]): (w, h).
        scale (float | tuple(float)): Scaling factor.

    Returns:
        tuple[int]: scaled size.
    """
    if isinstance(scale, (float, int)):
        scale = (scale, scale)
    w, h = size
    return int(w * float(scale[0]) + 0.5), int(h * float(scale[1]) + 0.5)

def rescale_size(old_size, scale, return_scale=False):
    """Calculate the new size to be rescaled to.

    Args:
        old_size (tuple[int]): The old size (w, h) of image.
        scale (float | tuple[int]): The scaling factor or maximum size.
            If it is a float number, then the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, then the image will
            be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image size.

    Returns:
        tuple[int]: The new rescaled image size.
    """
    w, h = old_size
    if isinstance(scale, (float, int)):
        if scale <= 0:
            raise ValueError(f'Invalid scale {scale}, must be positive.')
        scale_factor = scale
    elif isinstance(scale, tuple):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w),
                           max_short_edge / min(h, w))
    else:
        raise TypeError(
            f'Scale must be a number or tuple of int, but got {type(scale)}')

    new_size = _scale_size((w, h), scale_factor)

    if return_scale:
        return new_size, scale_factor
    else:
        return new_size

def imflip_(img, direction='horizontal'):
    """Inplace flip an image horizontally or vertically.

    Args:
        img (ndarray): Image to be flipped.
        direction (str): The flip direction, either "horizontal" or
            "vertical" or "diagonal".

    Returns:
        ndarray: The flipped image (inplace).
    """
    assert direction in ['horizontal', 'vertical', 'diagonal']
    if direction == 'horizontal':
        return cv2.flip(img, 1, img)
    elif direction == 'vertical':
        return cv2.flip(img, 0, img)
    else:
        return cv2.flip(img, -1, img)

def iminvert(img):
    """Invert (negate) an image.

    Args:
        img (ndarray): Image to be inverted.

    Returns:
        ndarray: The inverted image.
    """
    return np.full_like(img, 255) - img


def imnormalize_(img, mean, std, to_rgb=True):
    """Inplace normalize an image with mean and std.

    Args:
        img (ndarray): Image to be normalized.
        mean (ndarray): The mean to be used for normalize.
        std (ndarray): The std to be used for normalize.
        to_rgb (bool): Whether to convert to rgb.

    Returns:
        ndarray: The normalized image.
    """
    # cv2 inplace normalization does not accept uint8
    assert img.dtype != np.uint8
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace
    return img


def _init_lazy_if_proper(results, lazy):
    """Initialize lazy operation properly.

    Make sure that a lazy operation is properly initialized,
    and avoid a non-lazy operation accidentally getting mixed in.

    Required keys in results are "imgs" if "img_shape" not in results,
    otherwise, Required keys in results are "img_shape", add or modified keys
    are "img_shape", "lazy".
    Add or modified keys in "lazy" are "original_shape", "crop_bbox", "flip",
    "flip_direction", "interpolation".

    Args:
        results (dict): A dict stores data pipeline result.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    if 'img_shape' not in results:
        results['img_shape'] = results['imgs'][0].shape[:2]
    if lazy:
        if 'lazy' not in results:
            img_h, img_w = results['img_shape']
            lazyop = dict()
            lazyop['original_shape'] = results['img_shape']
            lazyop['crop_bbox'] = np.array([0, 0, img_w, img_h],
                                           dtype=np.float32)
            lazyop['flip'] = False
            lazyop['flip_direction'] = None
            lazyop['interpolation'] = None
            results['lazy'] = lazyop
    else:
        assert 'lazy' not in results, 'Use Fuse after lazy operations'


class SampleFrames:
    """Sample frames from the video.

    Required keys are "total_frames", "start_index" , added or modified keys
    are "frame_inds", "frame_interval" and "num_clips".

    Args:
        clip_len (int): Frames of each sampled output clip.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Default: 1.
        num_clips (int): Number of clips to be sampled. Default: 1.
        temporal_jitter (bool): Whether to apply temporal jittering.
            Default: False.
        twice_sample (bool): Whether to use twice sample when testing.
            If set to True, it will sample frames with and without fixed shift,
            which is commonly used for testing in TSM model. Default: False.
        out_of_bound_opt (str): The way to deal with out of bounds frame
            indexes. Available options are 'loop', 'repeat_last'.
            Default: 'loop'.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        start_index (None): This argument is deprecated and moved to dataset
            class (``BaseDataset``, ``VideoDatset``, ``RawframeDataset``, etc),
            see this: https://github.com/open-mmlab/mmaction2/pull/89.
        keep_tail_frames (bool): Whether to keep tail frames when sampling.
            Default: False.
    """

    def __init__(self,
                 clip_len,
                 frame_interval=1,
                 num_clips=1,
                 temporal_jitter=False,
                 twice_sample=False,
                 out_of_bound_opt='loop',
                 test_mode=False,
                 start_index=None,
                 keep_tail_frames=False):

        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.temporal_jitter = temporal_jitter
        self.twice_sample = twice_sample
        self.out_of_bound_opt = out_of_bound_opt
        self.test_mode = test_mode
        self.keep_tail_frames = keep_tail_frames
        assert self.out_of_bound_opt in ['loop', 'repeat_last']

        if start_index is not None:
            print('No longer support "start_index" in "SampleFrames", '
                          'it should be set in dataset class, see this pr: '
                          'https://github.com/open-mmlab/mmaction2/pull/89')

    def _get_train_clips(self, num_frames):
        """Get clip offsets in train mode.

        It will calculate the average interval for selected frames,
        and randomly shift them within offsets between [0, avg_interval].
        If the total number of frames is smaller than clips num or origin
        frames length, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval

        if self.keep_tail_frames:
            avg_interval = (num_frames - ori_clip_len + 1) / float(
                self.num_clips)
            if num_frames > ori_clip_len - 1:
                base_offsets = np.arange(self.num_clips) * avg_interval
                clip_offsets = (base_offsets + np.random.uniform(
                    0, avg_interval, self.num_clips)).astype(np.int)
            else:
                clip_offsets = np.zeros((self.num_clips, ), dtype=np.int)
        else:
            avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips

            if avg_interval > 0:
                base_offsets = np.arange(self.num_clips) * avg_interval
                clip_offsets = base_offsets + np.random.randint(
                    avg_interval, size=self.num_clips)
            elif num_frames > max(self.num_clips, ori_clip_len):
                clip_offsets = np.sort(
                    np.random.randint(
                        num_frames - ori_clip_len + 1, size=self.num_clips))
            elif avg_interval == 0:
                ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
                clip_offsets = np.around(np.arange(self.num_clips) * ratio)
            else:
                clip_offsets = np.zeros((self.num_clips, ), dtype=np.int)

        return clip_offsets

    def _get_test_clips(self, num_frames):
        """Get clip offsets in test mode.

        Calculate the average interval for selected frames, and shift them
        fixedly by avg_interval/2. If set twice_sample True, it will sample
        frames together without fixed shift. If the total number of frames is
        not enough, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in test mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) / float(self.num_clips)
        if num_frames > ori_clip_len - 1:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = (base_offsets + avg_interval / 2.0).astype(np.int)
            if self.twice_sample:
                clip_offsets = np.concatenate([clip_offsets, base_offsets])
        else:
            clip_offsets = np.zeros((self.num_clips, ), dtype=np.int)
        return clip_offsets

    def _sample_clips(self, num_frames):
        """Choose clip offsets for the video in a given mode.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices.
        """
        if self.test_mode:
            clip_offsets = self._get_test_clips(num_frames)
        else:
            clip_offsets = self._get_train_clips(num_frames)

        return clip_offsets

    def __call__(self, results):
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        total_frames = results['total_frames']

        clip_offsets = self._sample_clips(total_frames)
        frame_inds = clip_offsets[:, None] + np.arange(
            self.clip_len)[None, :] * self.frame_interval
        frame_inds = np.concatenate(frame_inds)

        if self.temporal_jitter:
            perframe_offsets = np.random.randint(
                self.frame_interval, size=len(frame_inds))
            frame_inds += perframe_offsets

        frame_inds = frame_inds.reshape((-1, self.clip_len))
        if self.out_of_bound_opt == 'loop':
            frame_inds = np.mod(frame_inds, total_frames)
        elif self.out_of_bound_opt == 'repeat_last':
            safe_inds = frame_inds < total_frames
            unsafe_inds = 1 - safe_inds
            last_ind = np.max(safe_inds * frame_inds, axis=1)
            new_inds = (safe_inds * frame_inds + (unsafe_inds.T * last_ind).T)
            frame_inds = new_inds
        else:
            raise ValueError('Illegal out_of_bound option.')

        start_index = results['start_index']
        frame_inds = np.concatenate(frame_inds) + start_index
        results['frame_inds'] = frame_inds.astype(np.int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = self.num_clips
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'frame_interval={self.frame_interval}, '
                    f'num_clips={self.num_clips}, '
                    f'temporal_jitter={self.temporal_jitter}, '
                    f'twice_sample={self.twice_sample}, '
                    f'out_of_bound_opt={self.out_of_bound_opt}, '
                    f'test_mode={self.test_mode})')
        return repr_str


class RawFrameDecode:
    """Load and decode frames with given indices.

    Required keys are "frame_dir", "filename_tmpl" and "frame_inds",
    added or modified keys are "imgs", "img_shape" and "original_shape".

    Args:
        io_backend (str): IO backend where frames are stored. Default: 'disk'.
        decoding_backend (str): Backend used for image decoding.
            Default: 'cv2'.
        kwargs (dict, optional): Arguments for FileClient.
    """

    def __init__(self, io_backend='disk', decoding_backend='cv2', **kwargs):
        self.io_backend = io_backend
        self.decoding_backend = decoding_backend
        self.kwargs = kwargs
        self.file_client = None

    def __call__(self, results):
        """Perform the ``RawFrameDecode`` to pick frames given indices.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """

        directory = results['frame_dir']
        filename_tmpl = results['filename_tmpl']
        modality = results['modality']

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        imgs = list()

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        offset = results.get('offset', 0)

        cache = {}
        for i, frame_idx in enumerate(results['frame_inds']):
            # Avoid loading duplicated frames
            if frame_idx in cache:
                if modality == 'RGB':
                    imgs.append(cp.deepcopy(imgs[cache[frame_idx]]))
                else:
                    imgs.append(cp.deepcopy(imgs[2 * cache[frame_idx]]))
                    imgs.append(cp.deepcopy(imgs[2 * cache[frame_idx] + 1]))
                continue
            else:
                cache[frame_idx] = i

            frame_idx += offset
            if modality == 'RGB':
                filepath = osp.join(directory, filename_tmpl.format(frame_idx))
                img_bytes = self.file_client.get(filepath)
                # Get frame with channel order RGB directly.
                cur_frame = imfrombytes(img_bytes, channel_order='rgb')
                imgs.append(cur_frame)
            elif modality == 'Flow':
                x_filepath = osp.join(directory,
                                      filename_tmpl.format('x', frame_idx))
                y_filepath = osp.join(directory,
                                      filename_tmpl.format('y', frame_idx))
                x_img_bytes = self.file_client.get(x_filepath)
                x_frame = imfrombytes(x_img_bytes, flag='grayscale')
                y_img_bytes = self.file_client.get(y_filepath)
                y_frame = imfrombytes(y_img_bytes, flag='grayscale')
                imgs.extend([x_frame, y_frame])
            else:
                raise NotImplementedError

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        # we resize the gt_bboxes and proposals to their real scale
        if 'gt_bboxes' in results:
            h, w = results['img_shape']
            scale_factor = np.array([w, h, w, h])
            gt_bboxes = results['gt_bboxes']
            gt_bboxes = (gt_bboxes * scale_factor).astype(np.float32)
            results['gt_bboxes'] = gt_bboxes
            if 'proposals' in results and results['proposals'] is not None:
                proposals = results['proposals']
                proposals = (proposals * scale_factor).astype(np.float32)
                results['proposals'] = proposals

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'io_backend={self.io_backend}, '
                    f'decoding_backend={self.decoding_backend})')
        return repr_str




class Resize:
    """Resize images to a specific size.

    Required keys are "img_shape", "modality", "imgs" (optional), "keypoint"
    (optional), added or modified keys are "imgs", "img_shape", "keep_ratio",
    "scale_factor", "lazy", "resize_size". Required keys in "lazy" is None,
    added or modified key is "interpolation".

    Args:
        scale (float | Tuple[int]): If keep_ratio is True, it serves as scaling
            factor or maximum size:
            If it is a float number, the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, the image will
            be rescaled as large as possible within the scale.
            Otherwise, it serves as (w, h) of output size.
        keep_ratio (bool): If set to True, Images will be resized without
            changing the aspect ratio. Otherwise, it will resize images to a
            given size. Default: True.
        interpolation (str): Algorithm used for interpolation:
            "nearest" | "bilinear". Default: "bilinear".
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self,
                 scale,
                 keep_ratio=True,
                 interpolation='bilinear',
                 lazy=False):
        if isinstance(scale, float):
            if scale <= 0:
                raise ValueError(f'Invalid scale {scale}, must be positive.')
        elif isinstance(scale, tuple):
            max_long_edge = max(scale)
            max_short_edge = min(scale)
            if max_short_edge == -1:
                # assign np.inf to long edge for rescaling short edge later.
                scale = (np.inf, max_long_edge)
        else:
            raise TypeError(
                f'Scale must be float or tuple of int, but got {type(scale)}')
        self.scale = scale
        self.keep_ratio = keep_ratio
        self.interpolation = interpolation
        self.lazy = lazy

    def _resize_imgs(self, imgs, new_w, new_h):
        return [
            imresize(
                img, (new_w, new_h), interpolation=self.interpolation)
            for img in imgs
        ]

    @staticmethod
    def _resize_kps(kps, scale_factor):
        return kps * scale_factor

    @staticmethod
    def _box_resize(box, scale_factor):
        """Rescale the bounding boxes according to the scale_factor.

        Args:
            box (np.ndarray): The bounding boxes.
            scale_factor (np.ndarray): The scale factor used for rescaling.
        """
        assert len(scale_factor) == 2
        scale_factor = np.concatenate([scale_factor, scale_factor])
        return box * scale_factor

    def __call__(self, results):
        """Performs the Resize augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """

        _init_lazy_if_proper(results, self.lazy)
        if 'keypoint' in results:
            assert not self.lazy, ('Keypoint Augmentations are not compatible '
                                   'with lazy == True')

        if 'scale_factor' not in results:
            results['scale_factor'] = np.array([1, 1], dtype=np.float32)
        img_h, img_w = results['img_shape']

        if self.keep_ratio:
            new_w, new_h = rescale_size((img_w, img_h), self.scale)
        else:
            new_w, new_h = self.scale

        self.scale_factor = np.array([new_w / img_w, new_h / img_h],
                                     dtype=np.float32)

        results['img_shape'] = (new_h, new_w)
        results['keep_ratio'] = self.keep_ratio
        results['scale_factor'] = results['scale_factor'] * self.scale_factor

        if not self.lazy:
            if 'imgs' in results:
                results['imgs'] = self._resize_imgs(results['imgs'], new_w,
                                                    new_h)
            if 'keypoint' in results:
                results['keypoint'] = self._resize_kps(results['keypoint'],
                                                       self.scale_factor)
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')
            lazyop['interpolation'] = self.interpolation

        if 'gt_bboxes' in results:
            assert not self.lazy
            results['gt_bboxes'] = self._box_resize(results['gt_bboxes'],
                                                    self.scale_factor)
            if 'proposals' in results and results['proposals'] is not None:
                assert results['proposals'].shape[1] == 4
                results['proposals'] = self._box_resize(
                    results['proposals'], self.scale_factor)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'scale={self.scale}, keep_ratio={self.keep_ratio}, '
                    f'interpolation={self.interpolation}, '
                    f'lazy={self.lazy})')
        return repr_str


class RandomCrop:
    """Vanilla square random crop that specifics the output size.

    Required keys in results are "img_shape", "keypoint" (optional), "imgs"
    (optional), added or modified keys are "keypoint", "imgs", "lazy"; Required
    keys in "lazy" are "flip", "crop_bbox", added or modified key is
    "crop_bbox".

    Args:
        size (int): The output size of the images.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self, size, lazy=False):
        if not isinstance(size, int):
            raise TypeError(f'Size must be an int, but got {type(size)}')
        self.size = size
        self.lazy = lazy

    @staticmethod
    def _crop_kps(kps, crop_bbox):
        return kps - crop_bbox[:2]

    @staticmethod
    def _crop_imgs(imgs, crop_bbox):
        x1, y1, x2, y2 = crop_bbox
        return [img[y1:y2, x1:x2] for img in imgs]

    @staticmethod
    def _box_crop(box, crop_bbox):
        """Crop the bounding boxes according to the crop_bbox.

        Args:
            box (np.ndarray): The bounding boxes.
            crop_bbox(np.ndarray): The bbox used to crop the original image.
        """

        x1, y1, x2, y2 = crop_bbox
        img_w, img_h = x2 - x1, y2 - y1

        box_ = box.copy()
        box_[..., 0::2] = np.clip(box[..., 0::2] - x1, 0, img_w - 1)
        box_[..., 1::2] = np.clip(box[..., 1::2] - y1, 0, img_h - 1)
        return box_

    def _all_box_crop(self, results, crop_bbox):
        """Crop the gt_bboxes and proposals in results according to crop_bbox.

        Args:
            results (dict): All information about the sample, which contain
                'gt_bboxes' and 'proposals' (optional).
            crop_bbox(np.ndarray): The bbox used to crop the original image.
        """
        results['gt_bboxes'] = self._box_crop(results['gt_bboxes'], crop_bbox)
        if 'proposals' in results and results['proposals'] is not None:
            assert results['proposals'].shape[1] == 4
            results['proposals'] = self._box_crop(results['proposals'],
                                                  crop_bbox)
        return results

    def __call__(self, results):
        """Performs the RandomCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)
        if 'keypoint' in results:
            assert not self.lazy, ('Keypoint Augmentations are not compatible '
                                   'with lazy == True')

        img_h, img_w = results['img_shape']
        assert self.size <= img_h and self.size <= img_w

        y_offset = 0
        x_offset = 0
        if img_h > self.size:
            y_offset = int(np.random.randint(0, img_h - self.size))
        if img_w > self.size:
            x_offset = int(np.random.randint(0, img_w - self.size))

        if 'crop_quadruple' not in results:
            results['crop_quadruple'] = np.array(
                [0, 0, 1, 1],  # x, y, w, h
                dtype=np.float32)

        x_ratio, y_ratio = x_offset / img_w, y_offset / img_h
        w_ratio, h_ratio = self.size / img_w, self.size / img_h

        old_crop_quadruple = results['crop_quadruple']
        old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]
        new_crop_quadruple = [
            old_x_ratio + x_ratio * old_w_ratio,
            old_y_ratio + y_ratio * old_h_ratio, w_ratio * old_w_ratio,
            h_ratio * old_h_ratio
        ]
        results['crop_quadruple'] = np.array(
            new_crop_quadruple, dtype=np.float32)

        new_h, new_w = self.size, self.size

        crop_bbox = np.array(
            [x_offset, y_offset, x_offset + new_w, y_offset + new_h])
        results['crop_bbox'] = crop_bbox

        results['img_shape'] = (new_h, new_w)

        if not self.lazy:
            if 'keypoint' in results:
                results['keypoint'] = self._crop_kps(results['keypoint'],
                                                     crop_bbox)
            if 'imgs' in results:
                results['imgs'] = self._crop_imgs(results['imgs'], crop_bbox)
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = x_offset * (lazy_right - lazy_left) / img_w
            right = (x_offset + new_w) * (lazy_right - lazy_left) / img_w
            top = y_offset * (lazy_bottom - lazy_top) / img_h
            bottom = (y_offset + new_h) * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        # Process entity boxes
        if 'gt_bboxes' in results:
            assert not self.lazy
            results = self._all_box_crop(results, results['crop_bbox'])

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}(size={self.size}, '
                    f'lazy={self.lazy})')
        return repr_str


class CenterCrop(RandomCrop):
    """Crop the center area from images.

    Required keys are "img_shape", "imgs" (optional), "keypoint" (optional),
    added or modified keys are "imgs", "keypoint", "crop_bbox", "lazy" and
    "img_shape". Required keys in "lazy" is "crop_bbox", added or modified key
    is "crop_bbox".

    Args:
        crop_size (int | tuple[int]): (w, h) of crop size.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self, crop_size, lazy=False):
        self.crop_size = _pair(crop_size)
        self.lazy = lazy
        if not is_tuple_of(self.crop_size, int):
            raise TypeError(f'Crop_size must be int or tuple of int, '
                            f'but got {type(crop_size)}')

    def __call__(self, results):
        """Performs the CenterCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)
        if 'keypoint' in results:
            assert not self.lazy, ('Keypoint Augmentations are not compatible '
                                   'with lazy == True')

        img_h, img_w = results['img_shape']
        crop_w, crop_h = self.crop_size

        left = (img_w - crop_w) // 2
        top = (img_h - crop_h) // 2
        right = left + crop_w
        bottom = top + crop_h
        new_h, new_w = bottom - top, right - left

        crop_bbox = np.array([left, top, right, bottom])
        results['crop_bbox'] = crop_bbox
        results['img_shape'] = (new_h, new_w)

        if 'crop_quadruple' not in results:
            results['crop_quadruple'] = np.array(
                [0, 0, 1, 1],  # x, y, w, h
                dtype=np.float32)

        x_ratio, y_ratio = left / img_w, top / img_h
        w_ratio, h_ratio = new_w / img_w, new_h / img_h

        old_crop_quadruple = results['crop_quadruple']
        old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]
        new_crop_quadruple = [
            old_x_ratio + x_ratio * old_w_ratio,
            old_y_ratio + y_ratio * old_h_ratio, w_ratio * old_w_ratio,
            h_ratio * old_h_ratio
        ]
        results['crop_quadruple'] = np.array(
            new_crop_quadruple, dtype=np.float32)

        if not self.lazy:
            if 'keypoint' in results:
                results['keypoint'] = self._crop_kps(results['keypoint'],
                                                     crop_bbox)
            if 'imgs' in results:
                results['imgs'] = self._crop_imgs(results['imgs'], crop_bbox)
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = left * (lazy_right - lazy_left) / img_w
            right = right * (lazy_right - lazy_left) / img_w
            top = top * (lazy_bottom - lazy_top) / img_h
            bottom = bottom * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        if 'gt_bboxes' in results:
            assert not self.lazy
            results = self._all_box_crop(results, results['crop_bbox'])

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}(crop_size={self.crop_size}, '
                    f'lazy={self.lazy})')
        return repr_str



class Flip:
    """Flip the input images with a probability.

    Reverse the order of elements in the given imgs with a specific direction.
    The shape of the imgs is preserved, but the elements are reordered.

    Required keys are "img_shape", "modality", "imgs" (optional), "keypoint"
    (optional), added or modified keys are "imgs", "keypoint", "lazy" and
    "flip_direction". Required keys in "lazy" is None, added or modified key
    are "flip" and "flip_direction". The Flip augmentation should be placed
    after any cropping / reshaping augmentations, to make sure crop_quadruple
    is calculated properly.

    Args:
        flip_ratio (float): Probability of implementing flip. Default: 0.5.
        direction (str): Flip imgs horizontally or vertically. Options are
            "horizontal" | "vertical". Default: "horizontal".
        flip_label_map (Dict[int, int] | None): Transform the label of the
            flipped image with the specific label. Default: None.
        left_kp (list[int]): Indexes of left keypoints, used to flip keypoints.
            Default: None.
        right_kp (list[ind]): Indexes of right keypoints, used to flip
            keypoints. Default: None.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """
    _directions = ['horizontal', 'vertical']

    def __init__(self,
                 flip_ratio=0.5,
                 direction='horizontal',
                 flip_label_map=None,
                 left_kp=None,
                 right_kp=None,
                 lazy=False):
        if direction not in self._directions:
            raise ValueError(f'Direction {direction} is not supported. '
                             f'Currently support ones are {self._directions}')
        self.flip_ratio = flip_ratio
        self.direction = direction
        self.flip_label_map = flip_label_map
        self.left_kp = left_kp
        self.right_kp = right_kp
        self.lazy = lazy

    def _flip_imgs(self, imgs, modality):
        _ = [imflip_(img, self.direction) for img in imgs]
        lt = len(imgs)
        if modality == 'Flow':
            # The 1st frame of each 2 frames is flow-x
            for i in range(0, lt, 2):
                imgs[i] = iminvert(imgs[i])
        return imgs

    def _flip_kps(self, kps, kpscores, img_width):
        kp_x = kps[..., 0]
        kp_x[kp_x != 0] = img_width - kp_x[kp_x != 0]
        new_order = list(range(kps.shape[2]))
        if self.left_kp is not None and self.right_kp is not None:
            for left, right in zip(self.left_kp, self.right_kp):
                new_order[left] = right
                new_order[right] = left
        kps = kps[:, :, new_order]
        if kpscores is not None:
            kpscores = kpscores[:, :, new_order]
        return kps, kpscores

    @staticmethod
    def _box_flip(box, img_width):
        """Flip the bounding boxes given the width of the image.

        Args:
            box (np.ndarray): The bounding boxes.
            img_width (int): The img width.
        """
        box_ = box.copy()
        box_[..., 0::4] = img_width - box[..., 2::4]
        box_[..., 2::4] = img_width - box[..., 0::4]
        return box_

    def __call__(self, results):
        """Performs the Flip augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)
        if 'keypoint' in results:
            assert not self.lazy, ('Keypoint Augmentations are not compatible '
                                   'with lazy == True')
            assert self.direction == 'horizontal', (
                'Only horizontal flips are'
                'supported for human keypoints')

        modality = results['modality']
        if modality == 'Flow':
            assert self.direction == 'horizontal'

        flip = np.random.rand() < self.flip_ratio

        results['flip'] = flip
        results['flip_direction'] = self.direction
        img_width = results['img_shape'][1]

        if self.flip_label_map is not None and flip:
            results['label'] = self.flip_label_map.get(results['label'],
                                                       results['label'])

        if not self.lazy:
            if flip:
                if 'imgs' in results:
                    results['imgs'] = self._flip_imgs(results['imgs'],
                                                      modality)
                if 'keypoint' in results:
                    kp = results['keypoint']
                    kpscore = results.get('keypoint_score', None)
                    kp, kpscore = self._flip_kps(kp, kpscore, img_width)
                    results['keypoint'] = kp
                    if 'keypoint_score' in results:
                        results['keypoint_score'] = kpscore
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Use one Flip please')
            lazyop['flip'] = flip
            lazyop['flip_direction'] = self.direction

        if 'gt_bboxes' in results and flip:
            assert not self.lazy and self.direction == 'horizontal'
            width = results['img_shape'][1]
            results['gt_bboxes'] = self._box_flip(results['gt_bboxes'], width)
            if 'proposals' in results and results['proposals'] is not None:
                assert results['proposals'].shape[1] == 4
                results['proposals'] = self._box_flip(results['proposals'],
                                                      width)

        return results

    def __repr__(self):
        repr_str = (
            f'{self.__class__.__name__}('
            f'flip_ratio={self.flip_ratio}, direction={self.direction}, '
            f'flip_label_map={self.flip_label_map}, lazy={self.lazy})')
        return repr_str


class Normalize:
    """Normalize images with the given mean and std value.

    Required keys are "imgs", "img_shape", "modality", added or modified
    keys are "imgs" and "img_norm_cfg". If modality is 'Flow', additional
    keys "scale_factor" is required

    Args:
        mean (Sequence[float]): Mean values of different channels.
        std (Sequence[float]): Std values of different channels.
        to_bgr (bool): Whether to convert channels from RGB to BGR.
            Default: False.
        adjust_magnitude (bool): Indicate whether to adjust the flow magnitude
            on 'scale_factor' when modality is 'Flow'. Default: False.
    """

    def __init__(self, mean, std, to_bgr=False, adjust_magnitude=False):
        if not isinstance(mean, Sequence):
            raise TypeError(
                f'Mean must be list, tuple or np.ndarray, but got {type(mean)}'
            )

        if not isinstance(std, Sequence):
            raise TypeError(
                f'Std must be list, tuple or np.ndarray, but got {type(std)}')

        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_bgr = to_bgr
        self.adjust_magnitude = adjust_magnitude

    def __call__(self, results):
        modality = results['modality']

        if modality == 'RGB':
            n = len(results['imgs'])
            h, w, c = results['imgs'][0].shape
            imgs = np.empty((n, h, w, c), dtype=np.float32)
            for i, img in enumerate(results['imgs']):
                imgs[i] = img

            for img in imgs:
                imnormalize_(img, self.mean, self.std, self.to_bgr)

            results['imgs'] = imgs
            results['img_norm_cfg'] = dict(
                mean=self.mean, std=self.std, to_bgr=self.to_bgr)
            return results
        if modality == 'Flow':
            num_imgs = len(results['imgs'])
            assert num_imgs % 2 == 0
            assert self.mean.shape[0] == 2
            assert self.std.shape[0] == 2
            n = num_imgs // 2
            h, w = results['imgs'][0].shape
            x_flow = np.empty((n, h, w), dtype=np.float32)
            y_flow = np.empty((n, h, w), dtype=np.float32)
            for i in range(n):
                x_flow[i] = results['imgs'][2 * i]
                y_flow[i] = results['imgs'][2 * i + 1]
            x_flow = (x_flow - self.mean[0]) / self.std[0]
            y_flow = (y_flow - self.mean[1]) / self.std[1]
            if self.adjust_magnitude:
                x_flow = x_flow * results['scale_factor'][0]
                y_flow = y_flow * results['scale_factor'][1]
            imgs = np.stack([x_flow, y_flow], axis=-1)
            results['imgs'] = imgs
            args = dict(
                mean=self.mean,
                std=self.std,
                to_bgr=self.to_bgr,
                adjust_magnitude=self.adjust_magnitude)
            results['img_norm_cfg'] = args
            return results
        raise NotImplementedError

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'mean={self.mean}, '
                    f'std={self.std}, '
                    f'to_bgr={self.to_bgr}, '
                    f'adjust_magnitude={self.adjust_magnitude})')
        return repr_str


class FormatShape:
    """Format final imgs shape to the given input_format.

    Required keys are "imgs", "num_clips" and "clip_len", added or modified
    keys are "imgs" and "input_shape".

    Args:
        input_format (str): Define the final imgs format.
        collapse (bool): To collpase input_format N... to ... (NCTHW to CTHW,
            etc.) if N is 1. Should be set as True when training and testing
            detectors. Default: False.
    """

    def __init__(self, input_format, collapse=False):
        self.input_format = input_format
        self.collapse = collapse
        if self.input_format not in ['NCTHW', 'NCHW', 'NCHW_Flow', 'NPTCHW']:
            raise ValueError(
                f'The input format {self.input_format} is invalid.')

    def __call__(self, results):
        """Performs the FormatShape formating.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if not isinstance(results['imgs'], np.ndarray):
            results['imgs'] = np.array(results['imgs'])
        imgs = results['imgs']
        # [M x H x W x C]
        # M = 1 * N_crops * N_clips * L
        if self.collapse:
            assert results['num_clips'] == 1

        if self.input_format == 'NCTHW':
            num_clips = results['num_clips']
            clip_len = results['clip_len']

            imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
            # N_crops x N_clips x L x H x W x C
            imgs = np.transpose(imgs, (0, 1, 5, 2, 3, 4))
            # N_crops x N_clips x C x L x H x W
            imgs = imgs.reshape((-1, ) + imgs.shape[2:])
            # M' x C x L x H x W
            # M' = N_crops x N_clips
        elif self.input_format == 'NCHW':
            imgs = np.transpose(imgs, (0, 3, 1, 2))
            # M x C x H x W
        elif self.input_format == 'NCHW_Flow':
            num_clips = results['num_clips']
            clip_len = results['clip_len']
            imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
            # N_crops x N_clips x L x H x W x C
            imgs = np.transpose(imgs, (0, 1, 2, 5, 3, 4))
            # N_crops x N_clips x L x C x H x W
            imgs = imgs.reshape((-1, imgs.shape[2] * imgs.shape[3]) +
                                imgs.shape[4:])
            # M' x C' x H x W
            # M' = N_crops x N_clips
            # C' = L x C
        elif self.input_format == 'NPTCHW':
            num_proposals = results['num_proposals']
            num_clips = results['num_clips']
            clip_len = results['clip_len']
            imgs = imgs.reshape((num_proposals, num_clips * clip_len) +
                                imgs.shape[1:])
            # P x M x H x W x C
            # M = N_clips x L
            imgs = np.transpose(imgs, (0, 1, 4, 2, 3))
            # P x M x C x H x W

        if self.collapse:
            assert imgs.shape[0] == 1
            imgs = imgs.squeeze(0)

        results['imgs'] = imgs
        results['input_shape'] = imgs.shape
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(input_format='{self.input_format}')"
        return repr_str


class Collect:
    """Collect data from the loader relevant to the specific task.

    This keeps the items in ``keys`` as it is, and collect items in
    ``meta_keys`` into a meta item called ``meta_name``.This is usually
    the last stage of the data loader pipeline.
    For example, when keys='imgs', meta_keys=('filename', 'label',
    'original_shape'), meta_name='img_metas', the results will be a dict with
    keys 'imgs' and 'img_metas', where 'img_metas' is a DataContainer of
    another dict with keys 'filename', 'label', 'original_shape'.

    Args:
        keys (Sequence[str]): Required keys to be collected.
        meta_name (str): The name of the key that contains meta infomation.
            This key is always populated. Default: "img_metas".
        meta_keys (Sequence[str]): Keys that are collected under meta_name.
            The contents of the ``meta_name`` dictionary depends on
            ``meta_keys``.
            By default this includes:

            - "filename": path to the image file
            - "label": label of the image file
            - "original_shape": original shape of the image as a tuple
                (h, w, c)
            - "img_shape": shape of the image input to the network as a tuple
                (h, w, c).  Note that images may be zero padded on the
                bottom/right, if the batch tensor is larger than this shape.
            - "pad_shape": image shape after padding
            - "flip_direction": a str in ("horiziontal", "vertival") to
                indicate if the image is fliped horizontally or vertically.
            - "img_norm_cfg": a dict of normalization information:
                - mean - per channel mean subtraction
                - std - per channel std divisor
                - to_rgb - bool indicating if bgr was converted to rgb
        nested (bool): If set as True, will apply data[x] = [data[x]] to all
            items in data. The arg is added for compatibility. Default: False.
    """

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'label', 'original_shape', 'img_shape',
                            'pad_shape', 'flip_direction', 'img_norm_cfg'),
                 meta_name='img_metas',
                 nested=False):
        self.keys = keys
        self.meta_keys = meta_keys
        self.meta_name = meta_name
        self.nested = nested

    def __call__(self, results):
        """Performs the Collect formating.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        data = {}
        for key in self.keys:
            data[key] = results[key]

        if len(self.meta_keys) != 0:
            meta = {}
            for key in self.meta_keys:
                meta[key] = results[key]
            data[self.meta_name] = meta
        if self.nested:
            for k in data:
                data[k] = [data[k]]

        return data

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'keys={self.keys}, meta_keys={self.meta_keys}, '
                f'nested={self.nested})')

class Compose:
    """Compose a data pipeline with a sequence of transforms.

    Args:
        transforms (list[dict | callable]): Either config
          dicts of transforms or transform objects.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
            dict: Transformed data.
        """
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data
