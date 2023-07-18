from matplotlib import pyplot as plt

from YOLO import YOLO, ImageResult
from datasets.YoloTransforms import *
from datasets.YoloDataset import YoloDataset
from yolov5.utils import LOGGER

def show_training_layer(img,stride,  imgsz):
    # stride = model.model[-1].stride[layer_idx]

    fig = plt.imshow(img)
    fig.axes.grid()
    fig.axes.set_xticks(np.arange(0, imgsz, stride))
    fig.axes.set_yticks(np.arange(0, imgsz, stride))

    def draw_box_center(box, c="r"):
        center_box = box_corner_to_center(box).cpu()
        center_x, center_y = center_box[0], center_box[1]
        fig.axes.plot(center_x, center_y, marker="o", c=c)

    for i, target in enumerate(targets):
        box = target[:4]
        cls = int(target[4])
        label = self.names[cls] + f" {i}"
        draw_box(fig.axes, box,
                 label, 'r')
        draw_box_center(box)

    positive_idx = gts_map > -1
    anchors = prediction[positive_idx].reshape(-1, len(self.names) + 5)
    gt_idxs = gts_map[positive_idx]
    for i, anchor in zip(gt_idxs, anchors):
        box = box_center_to_corner(anchor[:4])
        draw_box(fig.axes, box,
                 f"{int(i)} {round(float(anchor[4]), 4)}", 'g')
        draw_box_center(box.detach(), "g")


class DatasetEvaluator:
    def __init__(self, model: YOLO, dataset: YoloDataset, batch_size=1):
        self.model = model
        self.dataloader = dataset.get_dataloader(batch_size)
        if self.model.names is not None and self.model.names != dataset.names:
            LOGGER.Error("names must be equal")

    def get_iter(self, cutoff=None):
        size = cutoff if cutoff else len(self.dataloader)
        for i, data in enumerate(self.dataloader):
            if i == cutoff:
                break
            print(f"\r dataloader {i + 1}/{size}", end="")
            yield data

    def show(self, cutoff=None):
        for imgs, targets in self.get_iter(cutoff):
            for img, target in zip(imgs, targets):
                fig = plt.imshow(img.data)
                target.plot(fig.axes, self.model.names, ["r"] * len(self.model.names))
                plt.show()

    def detect(self, cutoff=None):
        self.model.eval_mode()
        for imgs, targets in self.get_iter(cutoff):
            preds = self.model(imgs)
            for img, target, pred in zip(imgs, targets, preds):
                fig = pred.plot(show=False)
                target.plot(fig.axes, self.model.names, ["r"] * len(self.model.names))
                plt.show()

    def show_training(self, cutoff=30):
        self.model.train_mode()



if __name__ == '__main__':
    mosaic = 0.0
    mixup = 1.0

    transforms = None
    # transforms = Transforms(RandomHorizontalFlip(), RandomHSV(), RandomRotate(10))
    # transforms = Transforms(RandomHorizontalFlip(), RandomHSV(), RandomCutImage(), proportions=[1.0, 0.2, 1.0])

    dataset = YoloDataset(path="datasets/train_fire.txt", input_size=640, augments=transforms, mosaic_prob=mosaic,
                          mixup_prob=mixup)


    model = YOLO("models/train/best.pt", ["fire", "smoke"], device_id=0)


    evaluator = DatasetEvaluator(model, dataset, 1)
    # evaluator.show(30)
    evaluator.detect(30)
