
import sys

from matplotlib import pyplot as plt

sys.path.append("..")

from models.yolo import yolov5, Detect


from utils.detect import *
from utils.box import box_corner_to_center, box_center_to_corner



new = False



class YOLO:
    def __init__(self, model, names, device=None, input_size=640, conf_thresh=0.5, nms_thresh=0.4):
        self.device = device
        self.input_size = input_size

        self.colors = rainbow_colors(len(names))

        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

        self.names = names

        self.model = yolov5(model).eval().float()
        self.model.to(device)
        Detect.new = new

        self.is_half = False
        self.timers = {}

    def show_training_anchors(self, layer_idx, targets, img, gts_map, prediction):

        stride = self.model.model[-1].stride[layer_idx]

        fig = plt.imshow(img)
        fig.axes.grid()
        fig.axes.set_xticks(np.arange(0, self.input_size, stride))
        fig.axes.set_yticks(np.arange(0, self.input_size, stride))

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

    def show_training(self, dataset, num_pics=-1):
        from utils.train import Assigner2, Assigner
        self.model.train()
        if new:
            assigner = Assigner2(self.input_size)
        else:
            assigner = Assigner(self.input_size)
        i = 0
        if num_pics < 0:
            num_pics = len(dataset)
        for Xs, targets in dataset:
            Xs = Xs.to(self.device)
            targets = [target.to(self.device) for target in targets]
            with torch.no_grad():
                predictions = self.model(Xs)
            assign_maps = assigner(self.model, predictions, targets)
            for layer_idx in range(3):
                prediction = predictions[layer_idx]
                gts_map = assign_maps[layer_idx]
                for batch_idx in range(len(targets)):
                    if i == num_pics * 3:
                        return 0
                    i += 1
                    print(f"\rpic {int(i / 3)}/{num_pics}", end="")
                    target = targets[batch_idx]
                    img = Xs[batch_idx].permute(1, 2, 0).cpu()
                    self.show_training_anchors(layer_idx, target, img, gts_map[batch_idx],
                                               prediction[batch_idx])
                    plt.show()

