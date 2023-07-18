import os
import shutil

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import math
import time
from copy import deepcopy

from . import LOGGER, Logger, check_dir, Timer
from .box import box_center_to_corner, box_corner_to_center, box_iou, box_giou


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
    if problems:
        return problems
    mask = loss < 0
    if mask.sum() == 0:
        return []
    error_log = f"The values in {name} are less than 0."
    loss_names = ["box", "obj", "cls"]
    for i, loss_name in enumerate(loss_names):
        value = float(loss[i])
        error_log += f"{loss_name}={value} "
    LOGGER.error(error_log)
    return ["<0"]


def get_cell_idx(x, y, grid_size):
    return 3 * (x + y * grid_size)


def get_cell_idxs(stride, grid_size, gt):
    idx_cells = []
    gt_x_cell = (gt[:, 0] / stride).long()
    gt_y_cell = (gt[:, 1] / stride).long()
    # print(stride,grid_size, gt_x_cell, gt_y_cell)
    idx_cell = get_cell_idx(gt_x_cell, gt_y_cell, grid_size)
    idx_cells.append(idx_cell)

    half = stride / 2
    x_mask = ((gt[:, 0] % stride) >= half).float()
    y_mask = ((gt[:, 1] % stride) >= half).float()

    idx_right = get_cell_idx(gt_x_cell + 1, gt_y_cell, grid_size) * x_mask
    idx_left = get_cell_idx(gt_x_cell - 1, gt_y_cell, grid_size) * (1 - x_mask)
    idx_bottom = get_cell_idx(gt_x_cell, gt_y_cell + 1, grid_size) * y_mask
    idx_top = get_cell_idx(gt_x_cell, gt_y_cell - 1, grid_size) * (1 - y_mask)
    idx_cells.append(idx_right + idx_left)
    idx_cells.append(idx_top + idx_bottom)

    return torch.stack(idx_cells, dim=1)


class Assigner:
    def __init__(self, input_size, ratio_thres=4.0):
        self.input_size = input_size
        self.ratio_thres = ratio_thres

    def __call__(self, model, predictions, targets):
        assign_maps = []
        detect = model.model[-1]
        for layer_idx in range(detect.nl):
            prediction = predictions[layer_idx]
            stride = detect.stride[layer_idx]
            anchors = detect.anchor_grid[layer_idx]
            assign_map = self.assign(prediction, targets, stride, anchors)
            assign_maps.append(assign_map)
        return assign_maps

    def assign(self, prediction, targets, stride, anchors):
        assign_map = torch.full(prediction.shape[:2], -1, device=prediction.device)

        grid_size = int(self.input_size / stride)
        anchors = anchors.view(1, -1, 2)
        batch_size = len(targets)

        for batch_idx in range(batch_size):
            gt = targets[batch_idx]
            if gt.shape[0] == 0:
                continue
            gt = box_corner_to_center(gt)
            anchor_cell_idxs = get_cell_idxs(stride, grid_size, gt).long()  # [num_gts,3]
            gt_wh = gt[:, 2:].unsqueeze(1)  # [num_gts 1,2]

            ratio1 = gt_wh / anchors
            ratio2 = anchors / gt_wh
            ratios = torch.max(ratio1, ratio2).max(dim=-1)[0]  # [num_gts,num_anchors]

            masks = ratios < self.ratio_thres
            similarest_anchor_idxs = torch.argmin(ratios, dim=1)
            masks[:, similarest_anchor_idxs] = True

            for gt_idx in range(gt.shape[0]):
                mask = masks[gt_idx]  # [num_anchors]
                idx_cell = anchor_cell_idxs[gt_idx]
                idx_cell = idx_cell[(0 <= idx_cell) & (idx_cell < prediction.shape[1])]
                assign_map[batch_idx, idx_cell + torch.nonzero(mask)] = gt_idx
        return assign_map


# class Assigner:
#     def __init__(self, input_size, ratio_thres=4.0):
#         self.input_size = input_size
#         self.ratio_thres = ratio_thres
#
#     def __call__(self, model, predictions, targets):
#         assign_maps = []
#         detect = model.model[-1]
#         for layer_idx in range(detect.nl):
#             prediction = predictions[layer_idx]
#             stride = detect.stride[layer_idx]
#             assign_map = self.assign(prediction, targets, stride)
#             assign_maps.append(assign_map)
#         return assign_maps
#
#     def assign(self, prediction, targets, stride):
#         device = prediction.device
#         assign_map = torch.full(prediction.shape[:2], -1, device=device)
#
#         grid_size = int(self.input_size / stride)
#         batch_size = len(targets)
#
#         for batch_idx in range(batch_size):
#             gt = targets[batch_idx]
#             if gt.shape[0] == 0:
#                 continue
#             gt = box_corner_to_center(gt)
#             grid_idxs = self.get_cell_idxs(stride, grid_size, gt[:, 0],
#                                            gt[:, 1]).long()  # [num_gts,num_anchors,num_cells]
#             positive_idxs = grid_idxs.view(-1)
#             gt_idxs = torch.repeat_interleave(torch.arange(0, gt.shape[0], device=device), 9, dim=0)
#             in_mask = (0 <= positive_idxs) & (positive_idxs < prediction.shape[1])
#             assign_map[batch_idx, positive_idxs[in_mask]] = gt_idxs[in_mask]
#
#         return assign_map
#
#
#     @staticmethod
#     def get_cell_idxs(stride, grid_size, x, y):
#
#         def get_cell_idx(xi, yi, grid_size):
#             idx = 3 * (xi + yi * grid_size)
#             return torch.stack((idx, idx + 1, idx + 2), dim=1)
#
#         gt_x_cell = (x / stride).long()
#         gt_y_cell = (y / stride).long()
#         idx_center = get_cell_idx(gt_x_cell, gt_y_cell, grid_size)
#
#         half = stride / 2
#         x_mask = ((x % stride) >= half).float().unsqueeze(1)
#         y_mask = ((y % stride) >= half).float().unsqueeze(1)
#
#         idx_right = get_cell_idx(gt_x_cell + 1, gt_y_cell, grid_size) * x_mask
#         idx_left = get_cell_idx(gt_x_cell - 1, gt_y_cell, grid_size) * (1 - x_mask)
#         idx_bottom = get_cell_idx(gt_x_cell, gt_y_cell + 1, grid_size) * y_mask
#         idx_top = get_cell_idx(gt_x_cell, gt_y_cell - 1, grid_size) * (1 - y_mask)
#
#         return torch.stack([idx_center, idx_right + idx_left, idx_top + idx_bottom], dim=2)


class Assigner1:
    def __init__(self, input_size, ratio_thres=4.0):
        self.input_size = input_size
        self.ratio_thres = ratio_thres

    def __call__(self, model, predictions, targets):
        assign_maps = []
        detect = model.model[-1]
        for layer_idx in range(detect.nl):
            prediction = predictions[layer_idx]
            stride = detect.stride[layer_idx]
            anchors = detect.anchor_grid[layer_idx]
            assign_map = self.assign(prediction, targets, stride, anchors)
            assign_maps.append(assign_map)
        return assign_maps

    def assign(self, prediction, targets, stride, anchors):
        device = prediction.device
        assign_map = torch.full(prediction.shape[:2], -1, device=device)

        grid_size = int(self.input_size / stride)
        anchors = anchors.view(1, -1, 2)
        batch_size = len(targets)

        for batch_idx in range(batch_size):
            gt = targets[batch_idx]
            if gt.shape[0] == 0:
                continue
            gt = box_corner_to_center(gt)
            grid_idxs = self.get_cell_idxs(stride, grid_size, gt[:, 0],
                                           gt[:, 1]).long()  # [num_gts,num_anchors,num_cells]
            gt_wh = gt[:, 2:].unsqueeze(1)  # [num_gts 1,2]

            ratio1 = gt_wh / anchors
            ratio2 = anchors / gt_wh
            scores = torch.max(ratio1, ratio2).max(dim=-1)[0]  # [num_gts,num_anchors]

            scores[:, torch.argmin(scores, dim=1)] = 0.0

            positive_mask = scores <= self.ratio_thres  # [num_gts,num_anchors]
            # scores = scores[positive_mask]  # [num_positives]
            positive_idxs = grid_idxs[positive_mask].view(-1)  # [num_positives, num_cells]
            # print(positive_idxs.shape)
            gt_idxs = torch.arange(0, gt.shape[0], device=device).view(-1, 1).repeat(1, 3)[
                positive_mask]  # [num_positives]
            # for cell_idx in positive_idxs.unique():
            #     mask = positive_idxs == cell_idx

            in_mask = (0 <= positive_idxs) & (positive_idxs < prediction.shape[1])
            assign_map[batch_idx, positive_idxs[in_mask]] = torch.repeat_interleave(gt_idxs, 3, dim=0)[in_mask]

        return assign_map

    @staticmethod
    def get_cell_idxs(stride, grid_size, x, y):

        def get_cell_idx(xi, yi, grid_size):
            idx = 3 * (xi + yi * grid_size)
            return torch.stack((idx, idx + 1, idx + 2), dim=1)

        gt_x_cell = (x / stride).long()
        gt_y_cell = (y / stride).long()
        idx_center = get_cell_idx(gt_x_cell, gt_y_cell, grid_size)

        half = stride / 2
        x_mask = ((x % stride) >= half).float().unsqueeze(1)
        y_mask = ((y % stride) >= half).float().unsqueeze(1)

        idx_right = get_cell_idx(gt_x_cell + 1, gt_y_cell, grid_size) * x_mask
        idx_left = get_cell_idx(gt_x_cell - 1, gt_y_cell, grid_size) * (1 - x_mask)
        idx_bottom = get_cell_idx(gt_x_cell, gt_y_cell + 1, grid_size) * y_mask
        idx_top = get_cell_idx(gt_x_cell, gt_y_cell - 1, grid_size) * (1 - y_mask)

        return torch.stack([idx_center, idx_right + idx_left, idx_top + idx_bottom], dim=2)


class Assigner2:
    def __init__(self, input_size, topk=13, iou_thresh=0.5):
        self.input_size = input_size
        self.topk = topk
        self.iou_thresh = iou_thresh

    def __call__(self, model, predictions, targets):
        assign_maps = []
        detect = model.model[-1]
        for layer_idx in range(detect.nl):
            prediction = predictions[layer_idx]
            assign_map = self.assign(prediction, targets)
            assign_maps.append(assign_map)
        return assign_maps

    def get_assigns(self, positive_mask, gt_idxs, scores, device):
        assigns = torch.full((int(positive_mask.sum()),), -1, device=device)
        for gt_idx in gt_idxs.unique():
            gt_idx_mask = gt_idxs == gt_idx  # (num_positives)

            if gt_idx_mask.sum() <= self.topk:
                assigns[gt_idx_mask] = gt_idx
                continue

            gt_scores = scores[gt_idx_mask]  # (num_positives_to_gt)
            _, pred_idxs = torch.topk(gt_scores, self.topk, dim=0)  # (self.topk), [0,num_positives_to_gt-1]
            pred_idxs = torch.nonzero(gt_idx_mask)[pred_idxs]  # (self.topk), [0,num_positive-1]
            assigns[pred_idxs] = gt_idx
        return assigns

    def assign(self, prediction, targets, conf_scores=False):
        assign_map = torch.full(prediction.shape[:2], -1, device=prediction.device)

        batch_size = len(targets)

        for img_idx in range(batch_size):
            pred_boxes = box_center_to_corner(prediction[img_idx])
            target = targets[img_idx]
            if target.shape[0] == 0:
                continue
            scores = box_iou(pred_boxes[:, :4], target[:, :4])  # [num_pred,num_target]
            if conf_scores:
                scores = scores ** 6.0 + prediction[img_idx, 4] ** 0.5

            scores[torch.argmax(scores, dim=0)] = 2.0

            b1 = pred_boxes[:, :4].unsqueeze(1)
            b2 = target[:, :4].unsqueeze(0)
            out_mask = (b1[..., 2] < b2[..., 0]) | (b1[..., 3] < b2[..., 1]) | (
                    b1[..., 0] > b2[..., 2]) | (b1[..., 1] > b2[..., 3])  # [num_pred,num_target]
            scores[out_mask] = 0.0

            scores, gt_idxs = torch.max(scores, dim=1)  # [num_pred]
            positive_mask = scores >= self.iou_thresh
            scores = scores[positive_mask]
            gt_idxs = gt_idxs[positive_mask]

            assigns = self.get_assigns(positive_mask, gt_idxs, scores, prediction.device)
            assign_map[img_idx, positive_mask] = assigns
        return assign_map

    def assign2(self, prediction, targets, layer_idx):
        assign_maps = []
        for img_idx in range(prediction.shape[0]):
            pred_img = prediction[img_idx]
            target_boxes = targets[img_idx]

            assign_map = torch.full(prediction.shape[1], -1)


class YoloLoss:
    hyperparameters = {
        "box": 0.05,
        "obj": 1.0,
        "cls": 0.3,
        "obj_pw": 1.0,
        "cls_pw": 1.0,
        "balance": [4.0, 1.0, 0.4]
    }

    def __init__(self, num_classes, input_size, assign_ratio_thres=4.0, iou_threshold=0.5, label_smoothing=0,
                 reduction="sum",
                 autoblance=False):
        self.num_classes = num_classes
        self.assigner = Assigner(input_size, ratio_thres=assign_ratio_thres)

        box_scaling = self.hyperparameters["box"]
        obj_scaling = (input_size / 640.0) ** 2 * self.hyperparameters["obj"]
        cls_scaling = num_classes / 80.0 * self.hyperparameters["cls"]
        self.scaling = torch.tensor([box_scaling, obj_scaling, cls_scaling])

        self.iou_threshold = iou_threshold
        self.label_smoothing = label_smoothing
        self.BCELoss = nn.BCELoss()
        self.reduction = reduction
        self.autobalance = autoblance

    def calc_loss(self, box_pred, box_label, obj_preds, positive_idxs, cls_preds, cls_labels):
        ious = box_giou(box_pred, box_label)
        loss_box = (1 - ious).mean()

        obj_labels = torch.zeros_like(obj_preds)
        obj_labels[positive_idxs] = ious.detach().clamp(0, 1)
        loss_obj = self.BCELoss(obj_preds, obj_labels)
        onehot = F.one_hot(cls_labels.long(), num_classes=self.num_classes).float()
        onehot = self.smooth_label(onehot, self.label_smoothing, self.num_classes)
        loss_cls = self.BCELoss(cls_preds, onehot)

        return torch.stack([loss_box, loss_obj, loss_cls])

    def __call__(self, model, predictions, targets):
        num_layers = model.model[-1].nl
        layer_scaling = num_layers / 3
        assign_maps = self.assigner(model, predictions, targets)
        loss = 0.0
        for layer_idx in range(num_layers):
            assign_map = assign_maps[layer_idx]
            prediction = predictions[layer_idx]
            layer_loss = self.forward(prediction, targets, assign_map, self.hyperparameters["balance"][layer_idx])
            loss += layer_loss
            if self.autobalance:
                self.hyperparameters["balance"][layer_idx] = self.hyperparameters["balance"][
                                                                 layer_idx] * 0.9999 + 0.0001 / loss[1].data
        loss *= self.scaling.to(loss.device).unsqueeze(0) * layer_scaling
        if self.reduction == "mean":
            return loss.mean(dim=0)
        if self.reduction == "sum":
            return loss.sum(dim=0)
        return loss

    def forward(self, prediction, targets, assign_maps, balance):
        loss = []
        batch_size = prediction.shape[0]
        for batch_idx in range(batch_size):
            target = targets[batch_idx]
            pred = prediction[batch_idx]
            if target.shape[0] == 0:
                # print("trained no obj")
                obj_pred = pred[:, 4]
                obj_loss = self.BCELoss(obj_pred, torch.zeros_like(obj_pred))
                loss.append(torch.stack([torch.zeros_like(obj_loss), obj_loss, torch.zeros_like(obj_loss)]))
                continue

            assign_map = assign_maps[batch_idx]
            positive_idx = assign_map > -1
            gt_idx = assign_map[positive_idx]

            box_pred = box_center_to_corner(pred[:, :4])
            box_label = target[:, :4]

            box_pred = box_pred[positive_idx]
            obj_pred = pred[:, 4]
            cls_pred = pred[:, 5:self.num_classes + 5][positive_idx]

            box_label = box_label[gt_idx]
            cls_label = target[:, 4][gt_idx]

            l = self.calc_loss(box_pred, box_label, obj_pred, positive_idx,
                               cls_pred, cls_label)
            loss.append(l)
        loss = torch.stack(loss)
        loss[1] *= balance
        return loss

    @staticmethod
    def smooth_label(cls_label, smoothing, num_classes):
        return cls_label * (1 - smoothing) + smoothing / num_classes



def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
        w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


class ComputeLoss:
    sort_obj_iou = False

    hyperparameters = {
        "box": 0.05,
        "obj": 1.0,
        "cls": 0.3,
        "obj_pw": 1.0,
        "cls_pw": 1.0,
        "balance": [1.0, 1.0, 1.0]
    }

    def __init__(self, model, input_size, autobalance=False):
        device = next(model.parameters()).device  # get model device
        self.input_size = input_size

        hyp = self.hyperparameters

        # BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hyp['cls_pw']], device=device))
        # BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hyp['obj_pw']], device=device))
        BCEcls = nn.BCELoss()
        BCEobj = nn.BCELoss()

        self.cp, self.cn = smooth_BCE(eps=hyp.get('label_smoothing', 0.0))

        # Focal loss
        # g = hyp['fl_gamma']  # focal loss gamma
        # if g > 0:
        #     BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        detect = model.model[-1]
        self.balance = {3: [4.0, 1.0, 0.4]}.get(detect.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(detect.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, hyp, autobalance
        self.na = detect.na  # number of anchors
        self.nc = detect.nc  # number of classes
        self.nl = detect.nl  # number of layers
        self.anchors = detect.anchors
        self.strides = detect.stride
        self.device = device

    def __call__(self, p, targets):  # predictions, targets

        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch

# if __name__ == '__main__':
#     a = torch.tensor([[1, 2, 3], [4, 5, 6]])
#     print(a.shape)
#     i = torch.tensor([0, 1])
#     b = a[torch.arange(0,i.shape[0]), i]
#     print(b)
