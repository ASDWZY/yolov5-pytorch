import torch
from torch.utils import data
import os
import cv2
import xmltodict
import numpy as np
from YoloTransforms import *
import re

FireSmokePerson_list = ["fire", "smoke", "person"]
FireSmokePerson_dict = {
    "fire": 0,
    "smoke": 1,
    "person": 2
}


class FireSmokeDataset(data.Dataset):
    def __init__(self, path="/home/sys205/GeneralData/数据集/data/fire_smoke_person", mode="train"):
        self.path = path
        with open(os.path.join(path, mode + "_list.txt")) as txt_file:
            self.datas = txt_file.read().split("\n")[:-1]
        self.size = len(self.datas)

    def __len__(self):
        return self.size

    def get_item(self, idx):
        data = self.datas[idx]
        img_dir, ann_dir = data.split(" ")
        img = cv2.imread(os.path.join(self.path, img_dir))[:, :, ::-1].astype(np.uint8)
        return img, os.path.join(self.path, ann_dir)

    def get_txt_str(self, idx):
        data = self.datas[idx]
        img_dir, ann_dir = data.split(" ")
        string = os.path.join(self.path, img_dir) + " ::"
        with open(os.path.join(self.path, ann_dir), "r+") as ann_file:
            stri = ann_file.read()
            try:
                objects = xmltodict.parse(stri)["annotation"]["object"]
            except:
                ann_file.seek(0)
                ret = re.search("xml version=.*>", stri)
                stri = stri[ret.start() + len(ret.group()):]
                ann_file.seek(0)
                ann_file.truncate()
                ann_file.write(stri)
                objects = xmltodict.parse(stri)["annotation"]["object"]
        if not isinstance(objects, (list, tuple)):
            objects = [objects]

        for object in objects:
            cls = FireSmokePerson_dict[object["name"]]
            box = object["bndbox"]
            xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
            string += f" {xmin},{ymin},{xmax},{ymax},{cls}"

        return string

    def write_ann(self, ann_dir, boxes):
        with open(ann_dir, "r+") as ann_file:
            objects = xmltodict.parse(ann_file.read())["annotation"]["object"]
            if not isinstance(objects, (list, tuple)):
                objects = [objects]
            for box in boxes:
                obj = {
                    "name": "person",
                    "bndbox": {"xmin": str(box[0]),
                               "ymin": str(box[1]),
                               "xmax": str(box[2]),
                               "ymax": str(box[3])
                               }
                }
                objects.append(obj)
            ann = {"annotation": {"object": objects}}
            string = xmltodict.unparse(ann)
            # print(string)
            ann_file.seek(0)
            ann_file.truncate()
            ann_file.write(string)

        ann_file.close()


class PersonDataset:
    def __init__(self, path='/home/sys205/GeneralData/数据集/data/VOCdevkit/VOC2012', mode='train'):
        txt_dir = os.path.join(path, 'ImageSets', "Main")
        self.annotations_dir = os.path.join(path, 'Annotations')
        self.img_dir = os.path.join(path, "JPEGImages")

        with open(os.path.join(txt_dir, mode + ".txt")) as txt_file:
            self.names = txt_file.read().split("\n")[:-1]

        self.size = len(self.names)

    def __len__(self):
        return self.size

    def get_txt_str(self, idx):
        name = self.names[idx]

        with open(os.path.join(self.annotations_dir, name + ".xml")) as ann_file:
            ann = xmltodict.parse(ann_file.read())["annotation"]
        # string = os.path.join(self.img_dir, ann["filename"]) + " ::"
        objects = ann["object"]
        if not isinstance(objects, (list, tuple)):
            objects = [objects]
        string = ""
        for object in objects:
            cls = object["name"]
            if cls != "person":
                continue
            cls = FireSmokePerson_dict[cls]
            box = object["bndbox"]
            xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
            string += f" {xmin},{ymin},{xmax},{ymax},{cls}"

        return os.path.join(self.img_dir, ann["filename"]) + " ::" + string if string else None


class FireSmokeDataset2(data.Dataset):
    def __init__(self, path="/home/sys205/GeneralData/数据集/data/FireSmoke", is_train=True):
        self.ann_path = os.path.join(path, "Annotations")
        self.img_path = os.path.join(path, "images")
        with open(os.path.join(path, "list.txt")) as txt_file:
            self.ann_names = txt_file.read().split("\n")[:-1]
        self.size = len(self.ann_names)
        if is_train:
            self.ann_names = self.ann_names[:int(self.size * 10 / 11)]  # 10:1
        else:
            self.ann_names = self.ann_names[int(self.size * 10 / 11):]
        self.size = len(self.ann_names)

    def __len__(self):
        return self.size

    def get_txt_str(self, idx):
        ann_name = self.ann_names[idx]

        with open(os.path.join(self.ann_path, ann_name + ".xml"), "r") as ann_file:
            stri = ann_file.read()
            ann = xmltodict.parse(stri)["Annotation"]
        img_filename = ann["filename"]
        string = os.path.join(self.img_path, img_filename + ".jpg") + " ::"
        try:
            objects = ann["object"]
        except:
            return None
        if not isinstance(objects, (list, tuple)):
            objects = [objects]

        for object in objects:
            cls = FireSmokePerson_dict[object["name"]]
            box = object["bndbox"]
            xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
            string += f" {xmin},{ymin},{xmax},{ymax},{cls}"

        return string


voc2012_list = ["aeroplane",
                "bicycle",
                "bird",
                "boat",
                "bottle",
                "bus",
                "car",
                "cat",
                "chair",
                "cow",
                "diningtable",
                "dog",
                "horse",
                "motorbike",
                "person",
                "pottedplant",
                "sheep",
                "sofa",
                "train",
                "tvmonitor"]


class VOC2012(data.Dataset):
    names_dict = {
        "aeroplane": 0,
        "bicycle": 1,
        "bird": 2,
        "boat": 3,
        "bottle": 4,
        "bus": 5,
        "car": 6,
        "cat": 7,
        "chair": 8,
        "cow": 9,
        "diningtable": 10,
        "dog": 11,
        "horse": 12,
        "motorbike": 13,
        "person": 14,
        "pottedplant": 15,
        "sheep": 16,
        "sofa": 17,
        "train": 18,
        "tvmonitor": 19
    }

    def __init__(self, path="/home/sys205/GeneralData/数据集/data/VOCdevkit/VOC2012", mode="train"):
        self.txt_dir = os.path.join(path, 'ImageSets', "Main")
        self.annotations_dir = os.path.join(path, 'Annotations')
        self.img_dir = os.path.join(path, "JPEGImages")

        with open(os.path.join(self.txt_dir, mode + ".txt")) as txt_file:
            self.names = txt_file.read().split("\n")[:-1]

        self.size = len(self.names)

    def __len__(self):
        return self.size

    def get_txt_str(self, idx):
        name = self.names[idx]
        with open(os.path.join(self.annotations_dir, name + ".xml")) as ann_flie:
            ann = xmltodict.parse(ann_flie.read())["annotation"]
        string = os.path.join(self.img_dir, ann["filename"]) + " ::"
        objects = ann["object"]
        if not isinstance(objects, (list, tuple)):
            objects = [objects]
        for obj in objects:
            try:
                cls = self.names_dict[obj["name"]]
            except:
                continue
            box = obj["bndbox"]
            xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
            string += f" {xmin},{ymin},{xmax},{ymax},{cls}"
        return string
