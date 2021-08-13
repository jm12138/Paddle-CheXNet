import os
import paddle
import numbers
import numpy as np
import paddle.nn as nn
import paddle.vision.transforms.functional as F

from paddle.metric import Metric
from sklearn.metrics import roc_auc_score
from collections.abc import Sequence


N_CLASSES = 14
CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]


def _is_numpy_(var):
    return isinstance(var, (np.ndarray, np.generic))


class AUROC(Metric):
    def __init__(self, num_classes, class_names, *args, **kwargs):
        super(AUROC, self).__init__(*args, **kwargs)
        self._num_classes = num_classes
        self._name = ['{}_{}'.format('AUROC', k)
                      for k in class_names] + ['AUROC_avg']
        self._AUROCs = []
        self._all_preds = []
        self._all_labels = []

    def update(self, preds, labels):
        """
        Update the auc curve with the given predictions and labels.
        Args:
            preds (numpy.array): An numpy array in the shape of
                (batch_size, 2), preds[i][j] denotes the probability of
                classifying the instance i into the class j.
            labels (numpy.array): an numpy array in the shape of
                (batch_size, 1), labels[i] is either o or 1,
                representing the label of the instance i.
        """
        if isinstance(labels, paddle.Tensor):
            labels = labels.numpy()
        elif not _is_numpy_(labels):
            raise ValueError("The 'labels' must be a numpy ndarray or Tensor.")

        if isinstance(preds, paddle.Tensor):
            preds = preds.numpy()
        elif not _is_numpy_(preds):
            raise ValueError("The 'preds' must be a numpy ndarray or Tensor.")

        self._all_preds.append(preds)
        self._all_labels.append(labels)

    def accumulate(self):
        """
        Return the area (a float score) under auc curve
        Return:
            float: the area under auc curve
        """
        preds = np.concatenate(self._all_preds, 0)
        labels = np.concatenate(self._all_labels, 0)
        AUROCs = []
        for i in range(self._num_classes):
            if labels[:, i].sum() == 0:
                return [0.0] * (self._num_classes + 1)
            AUROCs.append(roc_auc_score(labels[:, i], preds[:, i]))

        AUROC_avg = np.array(AUROCs).mean()
        return [auroc for auroc in AUROCs] + [AUROC_avg]

    def reset(self):
        """
        Reset states and result
        """
        self._all_preds = []
        self._all_labels = []

    def name(self):
        """
        Returns metric name
        """
        return self._name


def _get_image_size(img):
    # Returns (w, h) of tensor image
    if isinstance(img, (paddle.Tensor, np.ndarray)):
        return [img.shape[-1], img.shape[-2]]
    else:
        return img.size


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


def five_crop(img, size):
    """Crop the given image into four corners and the central crop.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    .. Note::
        This transform returns a tuple of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.

    Args:
        img (PIL Image or Tensor): Image to be cropped.
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).

    Returns:
       tuple: tuple (tl, tr, bl, br, center)
       Corresponding top left, top right, bottom left, bottom right and center crop.
    """
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    elif isinstance(size, (tuple, list)) and len(size) == 1:
        size = (size[0], size[0])

    if len(size) != 2:
        raise ValueError("Please provide only two dimensions (h, w) for size.")

    image_width, image_height = _get_image_size(img)
    crop_height, crop_width = size
    if crop_width > image_width or crop_height > image_height:
        msg = "Requested crop size {} is bigger than input size {}"
        raise ValueError(msg.format(size, (image_height, image_width)))

    tl = F.crop(img, 0, 0, crop_height, crop_width)
    tr = F.crop(img, 0, image_width - crop_width, crop_height, crop_width)
    bl = F.crop(img, image_height - crop_height, 0, crop_height, crop_width)
    br = F.crop(img, image_height - crop_height, image_width - crop_width,
                crop_height, crop_width)

    center = F.center_crop(img, [crop_height, crop_width])

    return tl, tr, bl, br, center


def ten_crop(img, size, vertical_flip=False):
    """Generate ten cropped images from the given image.
    Crop the given image into four corners and the central crop plus the
    flipped version of these (horizontal flipping is used by default).
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    .. Note::
        This transform returns a tuple of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.

    Args:
        img (PIL Image or Tensor): Image to be cropped.
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
        vertical_flip (bool): Use vertical flipping instead of horizontal

    Returns:
        tuple: tuple (tl, tr, bl, br, center, tl_flip, tr_flip, bl_flip, br_flip, center_flip)
        Corresponding top left, top right, bottom left, bottom right and
        center crop and same for the flipped image.
    """
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    elif isinstance(size, (tuple, list)) and len(size) == 1:
        size = (size[0], size[0])

    if len(size) != 2:
        raise ValueError("Please provide only two dimensions (h, w) for size.")

    first_five = five_crop(img, size)

    if vertical_flip:
        img = F.vflip(img)
    else:
        img = F.hflip(img)

    second_five = five_crop(img, size)
    return first_five + second_five


class TenCrop(nn.Layer):
    def __init__(self, size, vertical_flip=False):
        super().__init__()
        self.size = _setup_size(
            size,
            error_msg="Please provide only two dimensions (h, w) for size.")
        self.vertical_flip = vertical_flip

    def forward(self, img):

        return ten_crop(img, self.size, self.vertical_flip)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, vertical_flip={1})'.format(
            self.size, self.vertical_flip)


class Lambda:
    """Apply a user-defined lambda as a transform. This transform does not support torchscript.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """
    def __init__(self, lambd):
        if not callable(lambd):
            raise TypeError("Argument lambd should be callable, got {}".format(
                repr(type(lambd).__name__)))
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'