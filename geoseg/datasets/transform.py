import math
import numbers
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import random
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage import maximum_filter

#增强流水线
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

#SmartCrop内部使用它，从大图随机取训练区域
class RandomCrop(object):
    """
    Take a random crop from the image.
    First the image or crop size may need to be adjusted if the incoming image
    is too small...
    If the image is smaller than the crop, then:
         the image is padded up to the size of the crop
         unless 'nopad', in which case the crop size is shrunk to fit the image
    A random crop is taken such that the crop fits within the image.
    If a centroid is passed in, the crop must intersect the centroid.
    """
    def __init__(self, size=512, ignore_index=12, nopad=True):

        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.ignore_index = ignore_index
        self.nopad = nopad
        self.pad_color = (0, 0, 0)

    def __call__(self, img, mask, centroid=None):
        assert img.size == mask.size
        w, h = img.size
        # ASSUME H, W
        th, tw = self.size
        if w == tw and h == th:
            return img, mask

        if self.nopad:
            if th > h or tw > w:
                # Instead of padding, adjust crop size to the shorter edge of image.
                shorter_side = min(w, h)
                th, tw = shorter_side, shorter_side
        else:
            # Check if we need to pad img to fit for crop_size.
            if th > h:
                pad_h = (th - h) // 2 + 1
            else:
                pad_h = 0
            if tw > w:
                pad_w = (tw - w) // 2 + 1
            else:
                pad_w = 0
            border = (pad_w, pad_h, pad_w, pad_h)
            if pad_h or pad_w:
                img = ImageOps.expand(img, border=border, fill=self.pad_color)
                mask = ImageOps.expand(mask, border=border, fill=self.ignore_index)
                w, h = img.size

        if centroid is not None:
            # Need to insure that centroid is covered by crop and that crop
            # sits fully within the image
            c_x, c_y = centroid
            max_x = w - tw
            max_y = h - th
            x1 = random.randint(c_x - tw, c_x)
            x1 = min(max_x, max(0, x1))
            y1 = random.randint(c_y - th, c_y)
            y1 = min(max_y, max(0, y1))
        else:
            if w == tw:
                x1 = 0
            else:
                x1 = random.randint(0, w - tw)
            if h == th:
                y1 = 0
            else:
                y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class PadImage(object):
    def __init__(self, size=(512, 512), ignore_index=0):
        self.size = size
        self.ignore_index = ignore_index

    def __call__(self, img, mask):
        assert img.size == mask.size
        th, tw = self.size, self.size

        w, h = img.size

        if w > tw or h > th:
            wpercent = (tw / float(w))
            target_h = int((float(img.size[1]) * float(wpercent)))
            img, mask = img.resize((tw, target_h), Image.BICUBIC), mask.resize((tw, target_h), Image.NEAREST)

        w, h = img.size
        img = ImageOps.expand(img, border=(0, 0, tw - w, th - h), fill=0)
        mask = ImageOps.expand(mask, border=(0, 0, tw - w, th - h), fill=self.ignore_index)

        return img, mask


class RandomHorizontalFlip(object):

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if mask is not None:
            if random.random() < self.prob:
                return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(
                    Image.FLIP_LEFT_RIGHT)
            else:
                return img, mask
        else:
            if random.random() < self.prob:
                return img.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                return img


class RandomVerticalFlip(object):
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if mask is not None:
            if random.random() < self.prob:
                return img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(
                    Image.FLIP_TOP_BOTTOM)
            else:
                return img, mask
        else:
            if random.random() < self.prob:
                return img.transpose(Image.FLIP_TOP_BOTTOM)
            else:
                return img


class Resize(object):
    def __init__(self, size: tuple = (512,  512)):
        self.size = size  # size: (h, w)

    def __call__(self, img, mask):
        assert img.size == mask.size
        return img.resize(self.size, Image.BICUBIC), mask.resize(self.size, Image.NEAREST)

#随机缩放，让模型看到不同尺度目标
class RandomScale(object):
    def __init__(self, scale_list=[0.75, 1.0, 1.25], mode='value'):
        self.scale_list = scale_list
        self.mode = mode

    def __call__(self, img, mask):
        oh, ow = img.size
        #定义缩放比例
        scale_amt = 1.0
        #从给定值中选一个，离散尺度采样
        if self.mode == 'value':
            #np.random.choice()，是 NumPy 里面的随机采样函数。返回的是numpy数组，从self.scale_list随机选size：1个
            scale_amt = np.random.choice(self.scale_list, 1)
        #从给定值的范围中选一个，连续尺度采样
        elif self.mode == 'range':
            #random.uniform会在self.scale_list[0]和self.scale_list[-1]之间随机生成一个浮点数
            scale_amt = random.uniform(self.scale_list[0], self.scale_list[-1])
        h = int(scale_amt * oh)
        w = int(scale_amt * ow)
        #把训练图片按照Bicubic interpolation（双三次插值）算法缩放到(w, h)，标签按照Nearest Neighbor Interpolation（最近邻插值）缩放
        return img.resize((w, h), Image.BICUBIC), mask.resize((w, h), Image.NEAREST)


class ColorJitter(object):
    def __init__(self, brightness=0.5, contrast=0.5, saturation=0.5):
        if not brightness is None and brightness>0:
            self.brightness = [max(1-brightness, 0), 1+brightness]
        if not contrast is None and contrast>0:
            self.contrast = [max(1-contrast, 0), 1+contrast]
        if not saturation is None and saturation>0:
            self.saturation = [max(1-saturation, 0), 1+saturation]

    def __call__(self, img, mask=None):
        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])
        img = ImageEnhance.Brightness(img).enhance(r_brightness)
        img = ImageEnhance.Contrast(img).enhance(r_contrast)
        img = ImageEnhance.Color(img).enhance(r_saturation)
        if mask is None:
            return img
        else:
            return img, mask

#SmartCrop智能裁剪，就是智能的裁剪，对裁剪的图片加上了限制
#SmartCropV1限制了最大类别比例，就是要求裁剪后的图片里面至少包含两类且限制了最大类别比例，让每个patch尽可能包含多个语义类别
class SmartCropV1(object):
    def __init__(self, crop_size=512,
                 max_ratio=0.75,
                 ignore_index=12, nopad=False):
        self.crop_size = crop_size
        self.max_ratio = max_ratio
        self.ignore_index = ignore_index
        self.crop = RandomCrop(crop_size, ignore_index=ignore_index, nopad=nopad)

    def __call__(self, img, mask):
        assert img.size == mask.size
        count = 0
        while True:
            img_crop, mask_crop = self.crop(img.copy(), mask.copy())
            count += 1
            #np.unique去重并排序，加上return_counts=True后还会返回每个唯一值在原始数组中出现的次数（频数）
            labels, cnt = np.unique(np.array(mask_crop), return_counts=True)
            #去除ignore区域，这里用到了布尔索引，就是只取出方括号内为 True 的那些位置上的数据
            cnt = cnt[labels != self.ignore_index]
            #图片裁完后，至少包含两个类别，且最大类别不能超过总像素75%，这是为了对抗遥感图像极度类别不平衡
            if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < self.max_ratio:
                break
            #防止极端情况程序死循环
            if count > 10:
                break

        return img_crop, mask_crop

#SmartCropV2不仅限制了最大类别比例，还加上了指定关注重点关注某些类别，要求这些类别达到一定比例，适合对小目标类别进行强化
class SmartCropV2(object):
    def __init__(self, crop_size=512, num_classes=13,
                 class_interest=[2, 3],#指定关注类别
                 class_ratio=[0.1, 0.25],
                 max_ratio=0.75,
                 ignore_index=12, nopad=True):
        self.crop_size = crop_size
        self.num_classes = num_classes
        self.class_interest = class_interest
        self.class_ratio = class_ratio
        self.max_ratio = max_ratio
        self.ignore_index = ignore_index
        self.crop = RandomCrop(crop_size, ignore_index=ignore_index, nopad=nopad)

    def __call__(self, img, mask):
        assert img.size == mask.size
        count = 0
        while True:
            img_crop, mask_crop = self.crop(img.copy(), mask.copy())
            count += 1
            bins = np.array(range(self.num_classes + 1))
            class_pixel_counts, _ = np.histogram(np.array(mask_crop), bins=bins)
            cf = class_pixel_counts / (self.crop_size * self.crop_size)
            cf = np.array(cf)
            for c, f in zip(self.class_interest, self.class_ratio):
                if cf[c] > f:
                    break
            if np.max(cf) < 0.75 and np.argmax(cf) != self.ignore_index:
                break
            if count > 10:
                break

        return img_crop, mask_crop