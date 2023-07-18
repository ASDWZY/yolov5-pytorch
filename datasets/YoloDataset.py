import os
import random

import xmltodict
from torch.utils import data

import sys

sys.path.append("..")
from .YoloTransforms import *
from utils.image import Image
from utils.box import Boxes, NumpyBoxes


class BaseDataset(data.Dataset):
    names = []

    def __init__(self, root, img_dir, mode):
        self.root = root
        self.img_dir = os.path.join(root, img_dir)
        self.mode = mode
        self.datas = []

    def read(self, *path):
        if len(path) > 1:
            path = os.path.join(*path)
        with open(os.path.join(self.root, path)) as f:
            return f.read()

    def get_data(self, data):
        pass

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        return self.get_data(self.datas[item])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __str__(self):
        return ""

    def __repr__(self):
        return str(self)


class CrowdHuman(BaseDataset):
    def __init__(self, root, mode):
        super().__init__(root, "Images", mode)

        string = self.read(mode + ".odgt")


class YoloDataset(data.Dataset):

    def __init__(self, path, input_size, mosaic_prob=0.0, mixup_prob=0.0, names=None, augments=None):
        with open(path, encoding="gbk") as txt_file:
            datas = txt_file.readlines()
        # try:
        #     with open(path, encoding="gbk") as txt_file:
        #         self.datas = txt_file.readlines()
        # except:
        #     with open(path) as txt_file:
        #         self.datas = txt_file.readlines()
        self.names = datas[0].split(",")[:-1]
        if names is None:
            self._warn = True
            self.classes = [i for i in range(len(self.names))]
        else:
            self._warn = False
            self.classes = [i for i in range(len(self.names)) if self.names[i] in names]

        self._datas = datas[1:]

        self._mosaic_prob = mosaic_prob
        self._mixup_prob = mixup_prob

        self._mosaic = Mosaic(target_shape=(input_size, input_size))
        self._mixup = Mixup()

        self._augments = Transforms([Resize(input_size), LetterBox(input_size, input_size)])
        if augments:
            self._augments = Transforms([augments]) + self._augments

    def __len__(self):
        return len(self._datas)

    def _get_datas(self, idx):
        data = self._datas[idx]
        img_path, targets_str = data.split(" ::")
        # img = cv2.imread(img_path)[:, :, ::-1].astype(np.uint8)
        img = Image.read(img_path).rgb_data
        targets_str = targets_str.split(" ")[1:]
        targets = []
        for target_str in targets_str:
            x1, y1, x2, y2, cls = target_str.split(",")
            cls = int(eval(cls))
            if cls in self.classes:
                targets.append([float(x1), float(y1), float(x2), float(y2), cls])
        targets = torch.tensor(targets).reshape(-1, 5)
        return img, targets

    def _get_mosaic_datas(self, idx=None, num_datas=4):
        idxs = []
        if idx is not None:
            idxs.append(idx)
        while len(idxs) < num_datas:
            idxs.append(random.randint(0, len(self) - 1))
        imgs, targets = [], []
        for idx in idxs:
            img, target = self._get_datas(idx)
            imgs.append(img)
            targets.append(target)
        return self._mosaic(imgs, targets)

    def __getitem__(self, idx):
        if random.random() < self._mosaic_prob:
            img0, target0 = self._get_mosaic_datas(idx)
            if random.random() < self._mixup_prob:
                img1, target1 = self._get_mosaic_datas()
                return self._mixup([img0, img1], [target0, target1])
            return img0, target0

        img, targets = self._get_datas(idx)
        return self._augments(img, targets)

    def get_dataloader(self, batch_size, shuffle=True, num_workers=4):
        return data.DataLoader(self, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, drop_last=True,
                               collate_fn=self.collate_fn)

    @staticmethod
    def write(txt_path, dataset, names, cutoff=None):
        size = cutoff if cutoff is not None else len(dataset)
        with open(txt_path, "w") as txt_file:
            for name in names:
                txt_file.write(name + ",")
            txt_file.write("\n")
            for i in range(size):
                print(f"\r {i + 1}/{size}", end="")
                string = dataset.get_txt_str(i)
                if string is not None:
                    txt_file.write(string + "\n")
            txt_file.close()

    @staticmethod
    def collate_fn(batch):
        imgs = []
        targets = []
        for img, target in batch:
            imgs.append(Image(img))
            targets.append(Boxes(target, has_conf=False))
        return imgs, targets

    def close_mosaic(self):
        self._mosaic_prob = 0.0
