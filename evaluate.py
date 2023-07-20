import numpy as np
from matplotlib import pyplot as plt

from YOLO import YOLO, ImageResult
from datasets.YoloTransforms import *
from datasets.YoloDataset import YoloDataset
from yolov5.utils import LOGGER
from yolov5.utils.box import Boxes, squeeze_with_indices
from yolov5.utils.image import Image
from yolov5.utils.train import Assigner


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
        i = 0
        for imgs, targets in self.get_iter(cutoff):
            for img, target in zip(imgs, targets):
                i += 1
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

    def show_assignment_layer(self, img: Image, matched_targets, pred, stride):
        fig = img.letterbox(self.model.imgsz).plot(False)
        fig.axes.grid()
        fig.axes.set_xticks(np.arange(0, self.model.imgsz, int(stride)))
        fig.axes.set_yticks(np.arange(0, self.model.imgsz, int(stride)))

        if matched_targets is None:
            plt.show()
            return

        targets, bi, ai, xi, yi = matched_targets.split([5, 1, 1, 1, 1], dim=-1)
        targets = Boxes(targets, has_conf=False)
        xi, yi = xi.clamp(0, pred.shape[3] - 1), yi.clamp(0, pred.shape[2] - 1)

        names = self.model.names
        targets.plot(fig.axes, names, ["r"] * len(names))

        pred = Boxes(pred[bi, ai, yi, xi, :5].view(-1, 5), is_xywh=True)
        pred[:, 4] = pred.conf.sigmoid()

        pred.plot(fig.axes, names, ["g"] * len(names))

        plt.show()

    def show_assignment(self, cutoff=30):
        self.model.train_mode()
        assigner = Assigner()
        strides = self.model.get_strides()
        anchors = self.model.get_anchors()
        for imgs, targets in self.get_iter(cutoff):
            with torch.no_grad():
                preds, targets = self.model.train_process(imgs, targets)

            for layer_idx, pred in enumerate(preds):
                stride = strides[layer_idx]
                matched_targets = assigner(targets, stride, anchors[layer_idx])
                for bi, img in enumerate(imgs):
                    t = matched_targets[matched_targets[..., 5] == bi] if matched_targets is not None else None
                    self.show_assignment_layer(img, t, pred, stride)


if __name__ == '__main__':
    mosaic = 0.0
    mixup = 1.0

    transforms = None
    # transforms = Transforms(RandomHorizontalFlip(), RandomHSV(), RandomRotate(10))
    # transforms = Transforms(RandomHorizontalFlip(), RandomHSV(), RandomCutImage(), proportions=[1.0, 0.2, 1.0])

    dataset = YoloDataset(path="datasets/train_fire.txt", input_size=640, augments=transforms, mosaic_prob=mosaic,
                          mixup_prob=mixup)

    model = YOLO("models/train/best.pt", ["fire", "smoke"], device_id=-1)
    # model.reset_num_classes()
    # model.init_weights()

    evaluator = DatasetEvaluator(model, dataset, 1)
    # evaluator.show(30)
    evaluator.detect(30)
    # evaluator.show_assignment(5)
