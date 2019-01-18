from PIL import Image
import numpy as np
import math
import numbers
import random
from collections.abc import Iterable

from torch.nn.functional import pad

from torchvision import transforms
from torchvision.transforms import functional as F

def resize(img, size, interpolation=Image.BILINEAR, resize_small_dim=True):
    r"""Resize the input PIL Image to the given size.
    Args:
        img (PIL Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
        resize_smaller_dim (bool, optional): Whether to resize the larger or smaller
            dimension of image if int size parameter is given
    Returns:
        PIL Image: Resized image.
    """
    if not F._is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        w, h = img.size
        if resize_small_dim:
            if (w <= h and w == size) or (h <= w and h == size):
                return img
            if w < h:
                ow = size
                oh = int(size * h / w)
                return img.resize((ow, oh), interpolation)
            else:
                oh = size
                ow = int(size * w / h)
                return img.resize((ow, oh), interpolation)
        else: 
            if (w >= h and w == size) or (h >= w and h == size):
                return img
            if w > h:
                ow = size
                oh = int(size * h / w)
                return img.resize((ow, oh), interpolation)
            else:
                oh = size
                ow = int(size * w / h)
                return img.resize((ow, oh), interpolation)

    else:
        return img.resize(size[::-1], interpolation)


# PIL IMAGE TRANSFORMS
class Grayscale(transforms.Grayscale):
    def __call__(self, args):
        img, M = args

        img = super(Grayscale, self).__call__(img)
        return img, M

class Resize(transforms.Resize):
    def __init__(self, size, interpolation=Image.BILINEAR, resize_small_dim=True):
        self.size = size
        self.interpolation = interpolation
        self.resize_small_dim = resize_small_dim

    def __call__(self, args):
        img, M = args
        
        old_size = np.array([*img.size,1])
        # Apply regular torchvision Resize
        img = resize(img, self.size, self.interpolation, self.resize_small_dim)
        # update the transformation matrix
        new_size = np.array([*img.size,1])
        scale = new_size / old_size
        return img, np.diag(scale) @ M


class CenterCrop(transforms.CenterCrop):
    def __call__(self, args):
        img, M = args
        
        # Copied from F.center_crop
        output_size = self.size
        if isinstance(output_size, numbers.Number):
            output_size = (int(output_size), int(output_size))
        w, h = img.size
        th, tw = output_size
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        # Update transform matrix with translation
        M[:2,2] -= (j,i)
        return F.crop(img, i, j, th, tw), M

class RandomAffine(transforms.RandomAffine):
    def __call__(self, args):
        """
            img (PIL Image): Image to be transformed.

        Returns:
            PIL Image: Affine transformed image.
        """
        img, M = args
        ret = self.get_params(self.degrees, self.translate, 
                self.scale, self.shear, img.size)

        angle, translate, scale, shear = ret

        if not F._is_pil_image(img):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

        assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
            "Argument translate should be a list or tuple of length 2"

        assert scale >= 0.0, "Argument scale should be positive"

        output_size = img.size
        center = (img.size[0] * 0.5 + 0.5, img.size[1] * 0.5 + 0.5)
        matrix = F._get_inverse_affine_matrix(center, angle, translate, scale, shear)
        kwargs = {"fillcolor": self.fillcolor}
        img = img.transform(output_size, Image.AFFINE, matrix, self.resample, **kwargs)

        # Update transformation matrix
        inv_M = np.array([*matrix, 0,0,1]).reshape(3,3)
        M = np.linalg.inv(inv_M) @ M
        return img, M

class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __call__(self, args):
        img, M = args
        if random.random() < self.p:
            A = np.eye(3)
            # Flip the x axis and translate +x by the width of the image
            A[0,:] = (-1, 0, img.size[0])
            return F.hflip(img), A @ M
        return img, M


# TENSOR TRANSFORMS

class ToTensor(transforms.ToTensor):
    def __call__(self, args):
        img, M = args
        img = super(ToTensor, self).__call__(img)
        return img, M

class Normalize(transforms.Normalize):
    def __call__(self, args):
        img, M = args
        img = super(Normalize, self).__call__(img)
        return img, M

class ToPILImage(transforms.ToPILImage):
    def __call__(self, args):
        img, M = args
        img = super(ToPILImage, self).__call__(img)
        return img, M

class PadToSize():
    def __init__(self, size, mode='constant', value=0):
        self.size = (size, size) if isinstance(size, numbers.Number) else size
        self.mode = mode
        self.value = value

    def __call__(self, args):
        img, M = args
        ph, pw = self.size
        
        h,w = img.shape[1:]
        
        # If not evenly divisible, top and left pad have 1 extra pixel
        left_p = max(math.floor((pw - w)/2), 0)
        right_p = max(math.ceil((pw - w)/2), 0)
        
        top_p = max(math.floor((ph - h)/2), 0)
        bottom_p = max(math.ceil((ph - h)/2), 0)

        padding = (left_p, right_p, top_p, bottom_p)
        padded = pad(img, padding, self.mode, self.value)
        # This is equivalent to translating by the pad amount
        T = np.eye(3)
        T[:2,2] = (left_p, top_p)
        return padded, T @ M

    def __repr__(self):
        return self.__class__.__name__ + f'(size={self.size})'

# TargetTransforms

class NormalizePoints():
    def __init__(self, size):
        """ size (hight, width) of output image """
        self.size = (size,size) if isinstance(size, int) else size

    def __call__(self, points):
        points[0] /= self.size[1]
        points[1] /= self.size[0]
        points = np.clip(points, 0, 1)
        return points

class ToBBox():
    def __call__(self, points):
        return np.array([*points.min(1), *points.max(1)])

