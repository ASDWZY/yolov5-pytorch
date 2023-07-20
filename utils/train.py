import torch
from torch import nn
import torch.nn.functional as F

from . import LOGGER
from .box import Boxes


def check_value(x, name):
    problems = []
    if torch.isnan(x).sum() > 0:
        LOGGER.error(f"NaN in {name}")
        problems.append("nan")
    if torch.isinf(x).sum() > 0:
        LOGGER.error(f"INF in {name}")
        problems.append("inf")
    return problems


def check_loss(loss, name="loss"):
    problems = check_value(loss, name)
    if problems == []:
        if (loss < 0).sum() > 0:
            problems.append("<0")
            LOGGER.error(f"value<0 in {name}")
    if problems == []:
        return []

    error_log = "loss: "
    loss_names = ["box", "obj", "cls"]
    for i, loss_name in enumerate(loss_names):
        value = float(loss[i])
        error_log += f"{loss_name}={value} "
    LOGGER.error(error_log)
    return problems


class Assigner:
    def __init__(self, ratio_thresh=4.0, least1=True):
        self.ratio_thresh = ratio_thresh
        self.least1 = least1

    @staticmethod
    def _get_grid_targets(gt: Boxes, stride, ai):
        center = gt.center
        center_idx = (center / stride).long()
        half_mask = ((center % stride) > (stride / 2)).float().unsqueeze(-1)
        offsets = torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]], device=center.device)
        x_idxs = (center_idx + offsets[0]) * half_mask[:, 0] + (center_idx + offsets[1]) * (1 - half_mask[:, 0])
        y_idxs = (center_idx + offsets[2]) * half_mask[:, 1] + (center_idx + offsets[3]) * (1 - half_mask[:, 1])
        indices = torch.cat([center_idx, x_idxs.long(), y_idxs.long()], dim=0)
        # print(len(gt))
        # r = int(indices.shape[0] / len(gt))
        r = 3
        gt.to_xyxy()
        return torch.cat((gt.data.repeat(r, 1).long(), ai.view(-1, 1).repeat(r, 1), indices), dim=1)

    def __call__(self, gt: Boxes, stride, anchors):
        if len(gt) == 0:
            return torch.empty((0, 9), device=gt.device)
        r = (gt.wh.unsqueeze(0)) / (anchors.unsqueeze(1))  # shape(na, num_gts, 2)
        r = torch.max(r, 1 / r).max(2)[0]  # shape(na, num_gts)
        if self.least1:
            r[r.max(0)[1]] = 0.0
        pidx = torch.nonzero(r < self.ratio_thresh)  # shape(num_p, 2), ai,gi
        return self._get_grid_targets(gt[pidx[:, 1]], stride,
                                      pidx[:, 0])  # shape(num_p, 9) box,cls,bi,ai,yi,xi


class YoloLoss:
    hyperparameters = {
        "box": 0.05,
        "obj": 1.0,
        "cls": 0.3,
        "obj_pw": 1.0,
        "cls_pw": 1.0,
        "balance": [4.0, 1.0, 0.4]
    }

    def __init__(self, model, ratio_thresh=4.0, label_smoothing=0.0,
                 autoblance=False):
        num_classes = len(model.names)
        self.num_classes = num_classes
        self.assigner = Assigner(ratio_thresh)

        hyp = self.hyperparameters
        self.hyp = hyp.copy()

        box = hyp["box"]
        obj = (model.imgsz / 640.0) ** 2 * hyp["obj"]
        cls = num_classes / 80.0 * hyp["cls"]
        self.scaling = torch.tensor([box, obj, cls])

        self.cls_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hyp['cls_pw']], device=model.device))
        self.obj_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hyp['obj_pw']], device=model.device))

        self.label_smoothing = label_smoothing

        self.autobalance = autoblance

        self.strides = model.get_strides()

        self.anchors = model.get_anchors()

    def __call__(self, predictions, gt: Boxes, loss_name="loss"):
        total_loss = torch.zeros(3, device=gt.device)
        num_layers = 0
        for layer_idx, pred in enumerate(predictions):
            target_obj = torch.zeros(pred.shape[:-1], device=gt.device)
            balance = float(self.hyp["balance"][layer_idx])
            num_layers += 1

            matched_targets = self.assigner(gt, self.strides[layer_idx], self.anchors[layer_idx])
            if matched_targets.shape[0] > 0:
                target_box, target_cls, bi, ai, xi, yi = matched_targets.split([4, 1, 1, 1, 1, 1], dim=-1)
                xi, yi = xi.clamp(0, pred.shape[3] - 1), yi.clamp(0, pred.shape[2] - 1)
                pred_box, _, pred_cls = pred[bi, ai, yi, xi].view(-1, pred.shape[-1]).split([4, 1, self.num_classes],
                                                                                            dim=-1)
                iou = Boxes.giou(Boxes(pred_box, True, False), Boxes(target_box, has_conf=False))
                box_loss = (1.0 - iou).mean()
                target_obj[bi, ai, yi, xi] = iou.detach().clamp(0, 1).view(-1, 1)
                cls_loss = self.cls_loss(pred_cls, F.one_hot(target_cls.view(-1), num_classes=self.num_classes).float())

                obj_loss = self.obj_loss(pred[..., 4], target_obj)
                total_loss += (torch.stack([box_loss, obj_loss, cls_loss]) * balance)
            else:
                obj_loss = self.obj_loss(pred[..., 4], target_obj)
                total_loss[1] += obj_loss * balance
            if self.autobalance:
                self.hyp["balance"][layer_idx] = balance * 0.9999 + 0.0001 / float(obj_loss)
        total_loss *= self.scaling.to(gt.device) * (num_layers / 3)
        check_loss(total_loss, loss_name)
        return total_loss

    @staticmethod
    def smooth_label(cls_label, smoothing, num_classes):
        return cls_label * (1 - smoothing) + smoothing / num_classes
