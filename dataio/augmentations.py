import torch
import random
from kornia.enhance import adjust_contrast
from kornia.augmentation.functional import apply_adjust_contrast, apply_adjust_brightness


class Normalize(object):
    def __init__(self, min_val=0.0, max_val=1.0, eps=1e-6):
        self.min_val = min_val
        self.max_val = max_val
        self.eps = eps

    def __call__(self, x):
        return (self.max_val - self.min_val) * (x - x.min()) / (x.max() - x.min() + self.eps) + self.min_val


class CustomBrightness(object):
    def __init__(self, p, brightness=(1.0, 1.0)):
        self.p = p
        self.brightness = brightness

    def __call__(self, x):
        if random.random() <= self.p:
            brightness = torch.tensor(random.uniform(*self.brightness), dtype=torch.float32)
            return apply_adjust_brightness(x, {"brightness_factor": brightness})
        return x


class CustomContrast(object):
    def __init__(self, p, contrast=(1.0, 1.0)):
        self.p = p
        self.contrast = contrast

    def __call__(self, x):
        if random.random() <= self.p:
            brightness = torch.tensor(random.uniform(*self.contrast), dtype=torch.float32)
            return apply_adjust_contrast(x, {"contrast_factor": brightness})
        return x


class RandomContrast(object):
    def __init__(self, p, contrast):
        if isinstance(contrast, (int, float)):
            self.contrast = (float(contrast),)
        elif isinstance(contrast, (tuple, list)):
            if len(contrast) == 2:
                self.contrast = float(contrast[0]), float(contrast[1])
            else:
                raise ValueError("Contrast must be a 2-tuple or 2-list")
        self.p = p

    def __call__(self, x):
        if random.random() <= self.p:
            if len(self.contrast) == 2:
                contrast = random.uniform(*self.contrast)
            else:
                contrast = self.contrast[0]
            x = adjust_contrast(x, contrast)
        return x
