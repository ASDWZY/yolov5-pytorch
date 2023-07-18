import torch
import cv2
import random
import numpy as np
import math


class BaseTransform:
    def __call__(self, *args):
        return args

    def __str__(self):
        profile = ""
        for name, var in vars(self).items():
            profile += f"{name}={var}, "
        return f"{self.__class__.__name__}({profile[:-2]})"

    def __repr__(self):
        return str(self)

    @staticmethod
    def clip_boxes(targets, img_w, img_h, min_w=0, min_h=0):
        targets[:, [0, 2]] = targets[:, [0, 2]].clamp(0, img_w)
        targets[:, [1, 3]] = targets[:, [1, 3]].clamp(0, img_h)
        targets = targets[
            ((targets[:, 2] - targets[:, 0]) > min_w) & ((targets[:, 3] - targets[:, 1]) > min_h)]
        return targets

    @classmethod
    def affine_transform(cls, img, targets, matrix, target_shape, border=(128, 128, 128), min_w=0, min_h=0):
        img = cv2.warpAffine(img, matrix, target_shape, borderValue=border)
        if targets.shape[0] == 0:
            return img, targets
        boxes = torch.stack((targets[:, [0, 1]], targets[:, [0, 3]], targets[:, [2, 1]], targets[:, [2, 3]]), dim=1)
        ones = torch.ones((boxes.shape[0], boxes.shape[1], 1))
        boxes = torch.cat((boxes, ones), dim=-1)
        boxes = boxes @ matrix.T

        boxes_x, boxes_y = boxes[:, :, 0].view(-1, 4), boxes[:, :, 1].view(-1, 4)
        targets[:, 0] = torch.min(boxes_x, dim=1)[0]
        targets[:, 1] = torch.min(boxes_y, dim=1)[0]
        targets[:, 2] = torch.max(boxes_x, dim=1)[0]
        targets[:, 3] = torch.max(boxes_y, dim=1)[0]
        targets = cls.clip_boxes(targets, target_shape[0], target_shape[1], min_w, min_h)
        return img, targets

    @staticmethod
    def affine_shape(img_shape, matrix):
        img_h, img_w = img_shape
        box = torch.tensor([[0, 0, 1], [0, img_h, 1], [img_w, img_h, 1], [img_w, 0, 1]])
        box = box @ matrix.T
        x, y = box[:, 0], box[:, 1]

        return int(y.max() - y.min()), int(x.max() - x.min())

    @staticmethod
    def get_target_shape(*shape):
        if len(shape) == 1:
            return shape[0], shape[0]
        else:
            return shape[0], shape[1]

    @staticmethod
    def img2tensor(img):
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

    @staticmethod
    def resize(img, targets, target_shape):
        img_h, img_w = img.shape[:2]
        factor = min(target_shape[0] / img_w, target_shape[1] / img_h)
        new_img = cv2.resize(img, (0, 0), fx=factor, fy=factor)
        targets[:, :4] *= factor
        return new_img, targets

    @staticmethod
    def letterbox(img, targets, target_shape, border=None):
        w, h = target_shape
        if border is not None:
            new_img = np.zeros((h, w, img.shape[2]), dtype=np.uint8)
            new_img[:] = border
        else:
            new_img = np.random.randint(0, 255, (h, w, img.shape[2]), dtype=np.uint8)
        img_h, img_w = img.shape[:2]

        y_offset = (h - img_h) // 2
        x_offset = (w - img_w) // 2

        new_img[y_offset:y_offset + img_h, x_offset:x_offset + img_w] = img

        targets[:, [0, 2]] += x_offset
        targets[:, [1, 3]] += y_offset

        return new_img, targets


class ToTensor(BaseTransform):
    def __call__(self, img, targets):
        return self.img2tensor(img), targets


class Resize(BaseTransform):
    def __init__(self, *shape):
        self.target_shape = self.get_target_shape(*shape)

    def __call__(self, img, targets):
        return self.resize(img, targets, self.target_shape)


class LetterBox(BaseTransform):
    def __init__(self, *shape, border=None):
        self.target_shape = self.get_target_shape(*shape)
        self.border = border

    def __call__(self, img, targets):
        return self.letterbox(img, targets, self.target_shape, self.border)


class HorizontalFlip(BaseTransform):
    def __call__(self, img, targets):
        img = cv2.flip(img, 1, img)
        targets[:, [0, 2]] = img.shape[1] - targets[:, [2, 0]]
        return img, targets


class VerticalFlip(BaseTransform):
    def __call__(self, img, targets):
        img = cv2.flip(img, 0, img)
        targets[:, [1, 3]] = img.shape[0] - targets[:, [3, 1]]
        return img, targets


class CutImage(BaseTransform):
    def __init__(self, rect, min_w=0, min_h=0):
        self.rect = rect
        self.min_w = min_w
        self.min_h = min_h

    def __call__(self, img, targets):
        new_img = img[self.rect[1]:self.rect[3], self.rect[0]:self.rect[2]]
        targets[:, [0, 2]] -= self.rect[0]
        targets[:, [1, 3]] -= self.rect[1]
        targets = self.clip_boxes(targets, new_img.shape[1], new_img.shape[0], self.min_w, self.min_h)
        return new_img, targets


class AffineTransform(BaseTransform):
    def __init__(self, matrix, target_shape, border=(128, 128, 128)):
        self.matrix = matrix
        self.target_shape = target_shape
        self.border = border

    def __call__(self, img, targets):
        return self.affine_transform(img, targets, self.matrix, self.target_shape, self.border)


class Rotate(BaseTransform):
    def __init__(self, degree, target_shape=None):
        self.rad = degree * math.pi / 180.0
        self.target_shape = target_shape

    def __call__(self, img, targets):
        img_h, img_w = img.shape[:2]
        c, s = math.cos(self.rad), math.sin(self.rad)

        cx = img_w / 2
        cy = img_h / 2
        tx = (1 - c) * cx - s * cy
        ty = s * cx + (1 - c) * cy
        if self.target_shape:
            w, h = self.target_shape
        else:
            w = int(img_h * abs(s) + img_w * abs(c))
            h = int(img_h * abs(c) + img_w * abs(s))

        mat = np.array([[c, s, tx + w / 2 - cx],
                        [-s, c, ty + h / 2 - cy]])
        return self.affine_transform(img, targets, mat, (w, h))


class Shear(BaseTransform):
    def __init__(self, degree, target_shape=None):
        self.rad = degree * math.pi / 180.0
        self.target_shape = target_shape

    def __call__(self, img, targets):
        img_h, img_w = img.shape[:2]
        t = math.tan(self.rad)
        # c, s = math.cos(self.rad), math.sin(self.rad)
        mat = np.array([[1, t, 0.0],
                        [t, 1, 0.0]])
        shear = 0.7
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
        mat = S[:2]

        if self.target_shape:
            w, h = self.target_shape
        else:
            h, w = self.affine_shape(img.shape[:2], mat)
            print(h, w)

        mat = np.array([[1, t, 0.0],
                        [t, 1, 0.0]])
        return self.affine_transform(img, targets, mat, (w, h))


class NoiseBlocks(BaseTransform):
    def __init__(self, block_size=4, empty_size=8):
        self.block_size = block_size
        self.size = block_size + empty_size

    def __call__(self, img, targets):
        h, w = img.shape[:2]
        for y_offset in np.arange(0, h, self.size):
            for x_offset in np.arange(0, w, self.size):
                block_w = min(self.block_size, w - x_offset)
                block_h = min(self.block_size, h - y_offset)
                noise = np.random.randint(0, 255, (block_h, block_w, img.shape[2]))
                img[y_offset:y_offset + block_h, x_offset:x_offset + block_w] = noise
        return img, targets


def RandomHorizontalFlip(prob=0.5):
    return Transforms([HorizontalFlip()], probs=[prob])


def RandomVerticalFlip(prob=0.5):
    return Transforms([VerticalFlip()], probs=[prob])


def RandomFlip(horizontal_proportion=0.5, vertical_proportion=0.5):
    return Transforms([HorizontalFlip(), VerticalFlip()], probs=[horizontal_proportion, vertical_proportion])


class RandomHSV(BaseTransform):
    def __init__(self, hue=0.015, sat=0.7, val=0.4):
        self.hue = hue
        self.sat = sat
        self.val = val

    def __call__(self, img, targets):
        r = np.random.uniform(-1, 1, 3) * [self.hue, self.sat, self.val] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
        dtype = img.dtype

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        img = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

        return img, targets


class RandomSizeNoiseBlocks(BaseTransform):
    def __init__(self):
        self.min_size = 0


class RandomResize(BaseTransform):
    def __init__(self, minw, maxw, minh, maxh):
        self.minw = minw
        self.maxw = maxw
        self.minh = minh
        self.maxh = maxh

    def __call__(self, img, targets):
        img_h, img_w = img.shape[:2]
        fx = random.randint(self.minw, self.maxw) / img_w
        fy = random.randint(self.minh, self.maxh) / img_h
        fmix = min(fx, fy)
        img = cv2.resize(img, (0, 0), fx=fmix, fy=fmix)
        targets[:, :4] *= fmix

        return img, targets


class RandomResizeFactor(BaseTransform):
    def __init__(self, min_factor, max_factor):
        self.min_factor = min_factor
        self.max_factor = max_factor

    def __call__(self, img, targets):
        factor = random.uniform(self.min_factor, self.max_factor)
        img = cv2.resize(img, (0, 0), fx=factor, fy=factor)
        targets[:, :4] *= factor
        return img, targets


class RandomCutImage(BaseTransform):
    def __init__(self, *target_shape):
        self.target_shape = target_shape

    def __call__(self, img, targets):
        img_h, img_w = img.shape[:2]
        cutw, cuth = self.target_shape
        cutx = random.randint(0, img_w - cutw)
        cuty = random.randint(0, img_h - cuth)
        cut = CutImage((cutx, cuty, cutx + cutw, cuty + cuth))
        return cut(img, targets)


class RandomRotate(BaseTransform):
    def __init__(self, max_degree=10.0):
        self.max_degree = max_degree

    def __call__(self, img, targets):
        rotate = Rotate(random.uniform(-self.max_degree, self.max_degree))
        return rotate(img, targets)


class RandomErase(BaseTransform):
    def __init__(self, border=None, erase_min_prop=0.2, erase_max_prop=0.5, min_proportion=0.2):
        self.border = border
        self.erase_min_prop = erase_min_prop
        self.erase_max_prop = erase_max_prop
        self.min_proportion = min_proportion

    def __call__(self, img, targets):
        img_h, img_w = img.shape[:2]

        w, h = int(img_w * random.uniform(self.erase_min_prop, self.erase_max_prop)), int(
            img_h * random.uniform(self.erase_min_prop, self.erase_max_prop))
        minx, miny = random.randint(0, img_w - w), random.randint(0, img_h - h)
        if self.border:
            erase_area = np.zeros((h, w, img.shape[2]), dtype=np.uint8)
            erase_area[:] = self.border
        else:
            erase_area = np.random.randint(0, 255, (h, w, img.shape[2]), dtype=np.uint8)
        img[miny:miny + h, minx:minx + w] = erase_area

        x_in_mask = (minx <= targets[:, 0]) & (targets[:, 2] <= (minx + w))
        y_in_mask = (miny <= targets[:, 1]) & (targets[:, 3] <= (miny + h))
        out_mask = ~(x_in_mask & y_in_mask)
        if out_mask.sum() == 0:
            return img, torch.tensor([])
        targets = targets[out_mask]
        x_in_mask = x_in_mask[out_mask]
        y_in_mask = y_in_mask[out_mask]

        minw = (targets[:, 2] - targets[:, 0]) * self.min_proportion
        minh = (targets[:, 3] - targets[:, 1]) * self.min_proportion
        targets[x_in_mask & (miny <= targets[:, 1]) & (targets[:, 3] > (miny + h)), 1] = miny + h
        targets[x_in_mask & (miny > targets[:, 1]) & (targets[:, 3] <= (miny + h)), 3] = miny
        targets[(minx <= targets[:, 0]) & (targets[:, 2] > (minx + w)) & y_in_mask, 0] = minx + w
        targets[(minx > targets[:, 0]) & (targets[:, 2] <= (minx + w)) & y_in_mask, 2] = minx
        targets = targets[
            ((targets[:, 2] - targets[:, 0]) > minw) & ((targets[:, 3] - targets[:, 1]) > minh)]
        return img, targets


class Mosaic(BaseTransform):
    positions = [
        (-1, 0), (0, 0),
        (-1, -1), (0, -1)
    ]

    def __init__(self, target_shape, border=None, center_offset=0.15, min_proportion=0.2):
        assert center_offset < 0.5
        self.w, self.h = target_shape
        self.border = border
        self.center_offset = 0.5 - center_offset
        alpha, beta = 0.8, 1.0
        self.resize = RandomResize(int(self.w * alpha), int(self.w * beta), int(self.h * alpha), int(self.h * beta))
        self.min_proportion = min_proportion

    def clip(self, x, min, max):
        if x < min:
            return min
        if x > max:
            return max
        return x

    def __call__(self, imgs, targets):
        offset_x = random.uniform(self.center_offset, 1 - self.center_offset)
        offset_y = random.uniform(self.center_offset, 1 - self.center_offset)
        cutx, cuty = int(self.w * offset_x), int(self.h * offset_y)

        if self.border:
            mosaic_img = np.zeros((self.h, self.w, imgs[0].shape[2]), dtype=np.uint8)
            mosaic_img[:] = self.border
        else:
            mosaic_img = np.random.randint(0, 255, (self.h, self.w, imgs[0].shape[2]), dtype=np.uint8)
        mosaic_targets = []
        for idx in range(len(imgs)):
            img, target = imgs[idx], targets[idx]

            img, target = self.resize(img, target)
            img_h, img_w = img.shape[:2]
            img_xmin, img_ymin, img_xmax, img_ymax = 0, 0, img_w, img_h

            posx, posy = self.positions[idx]
            mosaic_xmin = cutx + posx * img_w
            mosaic_ymin = cuty + posy * img_h
            mosaic_xmax, mosaic_ymax = mosaic_xmin + img_w, mosaic_ymin + img_h

            if mosaic_xmin < 0:
                img_xmin = -mosaic_xmin
                mosaic_xmin = 0
            if mosaic_xmax > self.w:
                img_xmax = self.w - mosaic_xmin
                mosaic_xmax = self.w
            if mosaic_ymin < 0:
                img_ymin = -mosaic_ymin
                mosaic_ymin = 0
            if mosaic_ymax > self.h:
                img_ymax = self.h - mosaic_ymin
                mosaic_ymax = self.h

            mosaic_img[mosaic_ymin:mosaic_ymax, mosaic_xmin:mosaic_xmax] = img[img_ymin:img_ymax, img_xmin:img_xmax]
            minw = (target[:, 2] - target[:, 0]) * self.min_proportion
            minh = (target[:, 3] - target[:, 1]) * self.min_proportion

            target[:, [0, 2]] -= img_xmin
            target[:, [1, 3]] -= img_ymin
            target = self.clip_boxes(target, img_xmax - img_xmin, img_ymax - img_ymin, minw, minh)
            target[:, [0, 2]] += mosaic_xmin
            target[:, [1, 3]] += mosaic_ymin
            mosaic_targets.append(target)
        mosaic_targets = torch.cat(mosaic_targets, dim=0)
        return mosaic_img, mosaic_targets


class Mixup(BaseTransform):
    def __call__(self, imgs, targets):
        r = np.random.beta(32.0, 32.0)
        img = (imgs[0] * r + imgs[1] * (1 - r)).astype(np.uint8)
        new_targets = torch.cat((targets[0], targets[1]), dim=0)
        return img, new_targets


class MultiTransforms(list):
    def __init__(self, *args):
        # print(args)
        super().__init__(zip(*args))

    def __call__(self, *args):
        return args

    def __str__(self):
        profile = ""
        for arg in self:
            profile += str(arg) + ", "
        return f"{self.__class__.__name__}({profile[:-2]})"

    def __repr__(self):
        return str(self)

    def _new(self, datas):
        return self.__class__(*zip(*datas))

    def __add__(self, other):
        return self._new(super().__add__(other))


class Transforms(MultiTransforms):
    def __init__(self, transforms: list, probs=None):
        if probs is None:
            probs = [1.0] * len(transforms)
        super().__init__(transforms, probs)

    def __call__(self, *args):
        for transform, prob in self:
            if random.random() < prob:
                args = transform(*args)
                if args is None:
                    raise ValueError("args is None")
        return args


class Choice(MultiTransforms):
    def __init__(self, *transforms, proportions=None):
        if proportions:
            proportions = proportions
        else:
            proportions = [1.0 / len(transforms)] * len(transforms)
        s = sum(proportions)
        # if 1.0 - s > 1e-3:
        #     LOGGER.warning(f"proportions sum of {self} transform = {s},may cause None for result.")
        previous = 0
        for i in range(len(transforms)):
            proportions[i] += previous
            previous = proportions[i]
        self.sum = sum(proportions)
        # print(proportions, self.sum)
        # asd
        super().__init__(zip(transforms, proportions))

    def __call__(self, *args):
        choice = random.uniform(0.0, self.sum)
        previous = 0.0
        for transform, proportion in self:
            if previous < choice <= proportion:
                return transform(*args)
            previous = proportion
        return self[-1][0](*args)

    def __str__(self):
        profile = "Choose("
        for transform, proportion in self:
            profile += f" {transform}:{proportion}"
        return profile + ")"


if __name__ == '__main__':
    t = Transforms([NoiseBlocks(), RandomHSV()], probs=[0.3, 0.5])
    print(t)
    print(t + Transforms([RandomErase()]))
