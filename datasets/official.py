from yolov5.datasets.YoloDataset import YoloDataset
import os
from yolov5.utils import check_dir


def to_yolo(dataset: YoloDataset, root):
    check_dir(root, True)
    for i in range(len(dataset)):
        img, img_path, targets = dataset.get_datas(i)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]
        _, img_name = os.path.split(img_path)

        with open(os.path.join(root, img_name.replace(".jpg", ".txt")), "w") as f:
            for x1, y1, x2, y2, cls in targets:
                if cls != 0:
                    continue
                x = (x1 + x2) / 2 / img_w
                y = (y1 + y2) / 2 / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h
                f.write(f"{cls} {x} {y} {w} {h}\n")


if __name__ == '__main__':
    dataset = YoloDataset("val_fire.txt", 640)
    to_yolo(dataset, "/home/sys205/GeneralData/数据集/data/FIRESMOKE/val/labels")
