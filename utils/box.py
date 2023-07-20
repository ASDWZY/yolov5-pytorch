import numpy as np
import torch
from typing import Optional, List, Union
import cv2
from matplotlib import pyplot as plt


def all_equal(x: list):
    return len(set(x)) == 1


def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_corner_to_center(boxes):
    x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), dim=-1)
    return boxes


def box_center_to_corner(boxes):
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), dim=-1)
    return boxes


def box_iou(boxes1, boxes2):
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)

    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)

    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas


def box_giou(b1, b2, eps=1e-8):
    intersect_mins = torch.max(b1[..., :2], b2[..., :2])
    intersect_maxes = torch.min(b1[..., 2:], b2[..., 2:])
    intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = box_area(b1)
    b2_area = box_area(b2)
    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area / (union_area + eps)

    enclose_mins = torch.min(b1[..., :2], b2[..., :2])
    enclose_maxes = torch.max(b1[..., 2:], b2[..., 2:])
    enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))

    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    giou = iou - (enclose_area - union_area) / (enclose_area + eps)

    return giou


class Boxes:

    def __init__(self, data: Optional[torch.Tensor] = None, is_xywh=False, has_conf=True):
        if data is None:
            self.data = torch.empty((0, 4))
        else:
            self.data = data
            if self.data_size < 4:
                raise Exception("The data size must be greater than 3")
        self.is_xywh = is_xywh
        self.has_conf = has_conf
        self._cls_idx = 5 if self.has_conf else 4

    def __len__(self):
        return self.data.shape[0]

    def __str__(self):
        return f"{self.__class__.__name__}(data={self.data}, is_xywh={self.is_xywh}, has_conf={self.has_conf})"

    def __repr__(self):
        return str(self)

    def __getitem__(self, item):
        return self._new(self.data[item])

    def __setitem__(self, key, value):
        self.data[key] = value

    def __iter__(self):
        for d in self.data:
            yield self._new(d)

    def reshape(self, *shape):
        return self._new(self.data.reshape(*shape))

    @staticmethod
    def _stack(datas, axis):
        return torch.stack(datas, dim=axis)

    @staticmethod
    def _concat(datas, axis):
        return torch.cat(datas, dim=axis)

    @property
    def device(self):
        return self.data.device

    def to(self, device):
        self.data = self.data.to(device)
        return self

    @property
    def shape(self):
        return self.data.shape

    @property
    def data_size(self):
        return self.data.shape[-1]

    @property
    def dim(self):
        return len(self.data.shape)

    @property
    def boxes(self):
        return self.data[..., :4]

    @property
    def xywh(self):
        if self.is_xywh:
            return self.boxes
        x1, y1, x2, y2 = self.data[..., 0], self.data[..., 1], self.data[..., 2], self.data[..., 3]
        return self._stack(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), -1)

    @property
    def xyxy(self):
        if self.is_xywh:
            cx, cy, w, h = self.data[..., 0], self.data[..., 1], self.data[..., 2], self.data[..., 3]
            w2, h2 = w / 2, h / 2
            return self._stack((cx - w2, cy - h2, cx + w2, cy + h2), -1)
        return self.boxes

    @property
    def xx(self):
        if self.is_xywh:
            cx, w = self.data[..., 0], self.data[..., 2]
            w2 = w / 2
            return self._stack((cx - w2, cx + w2), -1)
        return self.data[..., [0, 2]]

    @property
    def yy(self):
        if self.is_xywh:
            cy, h = self.data[..., 1], self.data[..., 3]
            h2 = h / 2
            return self._stack((cy - h2, cy + h2), -1)
        return self.data[..., [1, 3]]

    @property
    def center(self):
        if self.is_xywh:
            return self.data[..., :2]
        x1, y1, x2, y2 = self.data[..., 0], self.data[..., 1], self.data[..., 2], self.data[..., 3]
        return self._stack(((x1 + x2) / 2, (y1 + y2) / 2), -1)

    @property
    def wh(self):
        if self.is_xywh:
            return self.data[..., 2:4]
        x1, y1, x2, y2 = self.data[..., 0], self.data[..., 1], self.data[..., 2], self.data[..., 3]
        return self._stack((x2 - x1, y2 - y1), -1)

    @property
    def cls_conf(self):
        return self.data[..., self._cls_idx:] if self.data_size > self._cls_idx else None

    @property
    def cls(self):
        return self.data[..., self._cls_idx] if self.data_size > self._cls_idx else None

    @property
    def unique_classes(self):
        return self.cls.unique()

    @property
    def conf(self):
        if self.has_conf and self.data_size > 4:
            return self.data[..., 4]

    @property
    def track_id(self):
        return self.data[..., self._cls_idx + 1] if self.data_size > self._cls_idx + 1 else None

    def _new(self, data=None):
        return self.__class__(data, self.is_xywh, self.has_conf)

    def get_cls_boxes(self, cls):
        return self._new(self.data[self.cls == cls])

    def from_cls_conf(self):
        cls_conf = self.cls_conf
        if cls_conf is not None:
            return cls_conf.max(-1)

    def copy(self):
        return self._new(self.data.clone())

    def add_attrs(self, *attrs):
        return self._new(self._concat((self.data, *attrs), -1))

    def to_xywh(self):
        if not self.is_xywh:
            self.is_xywh = True
            x1, y1, x2, y2 = self.data[..., 0], self.data[..., 1], self.data[..., 2], self.data[..., 3]
            self.data[..., :4] = self._stack(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), -1)

    def to_xyxy(self):
        if self.is_xywh:
            self.is_xywh = False
            cx, cy, w, h = self.data[..., 0], self.data[..., 1], self.data[..., 2], self.data[..., 3]
            w2, h2 = w / 2, h / 2
            self.data[..., :4] = self._stack((cx - w2, cy - h2, cx + w2, cy + h2), -1)

    def select_by_conf(self, conf_thresh):
        if self.has_conf:
            boxes = self.data[self.conf > conf_thresh]
            if boxes.shape[0] == 0:
                return self._new()
            cls_score, classes = boxes[..., self._cls_idx:].max(dim=-1)
            conf = boxes[..., 4] * cls_score
        else:
            boxes = self.data.clone()
            conf, classes = boxes[..., self._cls_idx:].max(dim=-1)
        mask = conf >= conf_thresh
        conf = conf[mask].unsqueeze(-1)
        if conf.shape[0] == 0:
            return self._new()
        return self.__class__(self._concat((boxes[mask][:, :4], conf, classes[mask].unsqueeze(-1)), -1),
                              is_xywh=self.is_xywh)

    def clip(self, w, h=None):
        if h is None:
            self.data[..., :4] = torch.clamp(self.data[..., :4], min=0, max=w)
        else:
            self.data[..., [0, 2]] = torch.clamp(self.data[..., [0, 2]], min=0, max=w)
            self.data[..., [1, 3]] = torch.clamp(self.data[..., [1, 3]], min=0, max=h)

    def nms(self, nms_thresh):
        if self.dim != 2:
            raise NotImplementedError("Dim must be 2")
        if self.conf is None:
            raise IndexError("boxes to nms must have confidences")
        boxes = self.xyxy
        ranking_idxs = torch.sort(self.conf, dim=0, descending=True)[1]
        keep = []
        while ranking_idxs.shape[0] > 0:
            i = int(ranking_idxs[0])
            keep.append(i)
            if ranking_idxs.shape[0] == 1:
                break
            iou = self.iou_mat(self.__class__(boxes[i].view(1, 4)), self.__class__(boxes[ranking_idxs[1:]])).view(-1)
            ranking_idxs = ranking_idxs[1:][iou <= nms_thresh]
        return self[torch.tensor(keep, device=self.device)]

    def letterbox(self, img_shape, imgsz):
        img_h, img_w = img_shape
        factor = imgsz / max(img_shape)
        self.data[..., :4] *= factor
        self.data[..., [0, 2]] += (imgsz - factor * img_w) // 2
        self.data[..., [1, 3]] += (imgsz - factor * img_h) // 2

    def reverse_letterbox(self, img_shape, imgsz):
        img_h, img_w = img_shape
        factor = imgsz / max(img_shape)
        self.to_xyxy()
        self.data[..., [0, 2]] -= (imgsz - factor * img_w) // 2
        self.data[..., [1, 3]] -= (imgsz - factor * img_h) // 2
        self.data[..., :4] /= factor

    def _get_box_label(self, box, names, colors, box_color=(0, 255, 0)):
        label = ""
        if self.data_size > self._cls_idx + 1:
            label = f"id={int(box[self._cls_idx + 1])} "
        if self.data_size > self._cls_idx:
            cls = int(box[self._cls_idx])
            try:
                label += names[cls]
                box_color = colors[cls]
            except IndexError:
                print(f"class out of names in draw : class={cls}, names={names}")
        if self.has_conf and self.data_size > 4:
            label += "%.2f" % float(box[4])
        return label, box_color

    def draw(self, img, names, colors, label_color=(255, 255, 255), thickness: int = 2,
             font=cv2.FONT_HERSHEY_SIMPLEX,
             font_scale: float = 1.0):
        if self.dim != 2:
            raise NotImplementedError("Dim must be 2")
        assert len(names) == len(colors)
        self.to_xyxy()
        for box in self.data:
            lt, rb = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            label, box_color = self._get_box_label(box, names, colors)
            cv2.rectangle(img, lt, rb, box_color, thickness)
            text_width, text_height = cv2.getTextSize(label, font, fontScale=font_scale, thickness=2)[0]
            cv2.rectangle(img, lt, (lt[0] + text_width, lt[1] - text_height), box_color, -1)
            cv2.putText(img, label, lt, font, font_scale, label_color, 2, lineType=cv2.LINE_AA)

    def plot(self, axes, names, colors, label_color="w", thickness: int = 2):
        if self.dim != 2:
            raise NotImplementedError("Dim must be 2")
        assert len(names) == len(colors)
        self.to_xyxy()
        for box in self.data:
            label, box_color = self._get_box_label(box, names, colors, "g")
            rect = plt.Rectangle(
                xy=(int(box[0]), int(box[1])), width=int(box[2] - box[0]), height=int(box[3] - box[1]),
                fill=False, edgecolor=box_color, linewidth=thickness)
            axes.add_patch(rect)
            axes.text(rect.xy[0], rect.xy[1], label,
                      va='center', ha='center', fontsize=9, color=label_color,
                      bbox=dict(facecolor=box_color, lw=0))

    @classmethod
    def concat(cls, x, axis=0):
        ret_conf = all_equal([y.has_conf for y in x])
        ret_xywh = all_equal([y.is_xywh for y in x])
        if ret_conf and ret_xywh:
            return cls(cls._concat([y.data for y in x], axis), is_xywh=x[0].is_xywh, has_conf=x[0].has_conf)
        raise ValueError("has_conf and is_xywh of boxes for concatenation must be equal.")

    @property
    def box_area(self):
        boxes = self.xyxy
        return (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])

    @staticmethod
    def iou(boxes1, boxes2, eps=1e-8):
        b1, b2 = boxes1.xyxy, boxes2.xyxy
        intersect_mins = torch.max(b1[..., :2], b2[..., :2])
        intersect_maxes = torch.min(b1[..., 2:], b2[..., 2:])
        intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        union_area = boxes1.box_area + boxes2.box_area - intersect_area
        return intersect_area / (union_area + eps)

    @staticmethod
    def iou_mat(b1, b2, eps=1e-8):
        if not b1.dim == b2.dim == 2:
            raise NotImplementedError("iou_mat only for boxes dim=2")
        boxes1, boxes2 = b1.xyxy, b2.xyxy

        inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)

        inter_areas = inters[:, :, 0] * inters[:, :, 1]
        union_areas = b1.box_area[:, None] + b2.box_area - inter_areas
        return inter_areas / (union_areas + eps)

    @staticmethod
    def giou(boxes1, boxes2, eps=1e-8):
        b1, b2 = boxes1.xyxy, boxes2.xyxy
        intersect_mins = torch.max(b1[..., :2], b2[..., :2])
        intersect_maxes = torch.min(b1[..., 2:], b2[..., 2:])
        intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        union_area = boxes1.box_area + boxes2.box_area - intersect_area
        iou = intersect_area / (union_area + eps)

        enclose_mins = torch.min(b1[..., :2], b2[..., :2])
        enclose_maxes = torch.max(b1[..., 2:], b2[..., 2:])
        enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(enclose_maxes))

        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
        return iou - (enclose_area - union_area) / (enclose_area + eps)


class NumpyBoxes(Boxes):
    def __init__(self, data: Optional[np.ndarray] = None, is_xywh=False, has_conf=True):
        super().__init__(is_xywh=is_xywh, has_conf=has_conf)
        self.data = data if data is not None else np.zeros((0, 4))
        if self.data_size < 4:
            raise Exception("The data size must be greater than 3")

    @staticmethod
    def _stack(datas, axis):
        return np.stack(datas, axis=axis)

    @staticmethod
    def _concat(datas, axis):
        return np.concatenate(datas, axis=axis)

    @property
    def unique_classes(self):
        return np.unique(self.cls)

    def copy(self):
        return self._new(self.data.copy())

    def from_cls_conf(self):
        cls_conf = self.cls_conf
        if cls_conf is not None:
            return cls_conf.max(-1), np.argmax(cls_conf, axis=-1)

    def select_by_conf(self, conf_thresh):
        if self.has_conf:
            boxes = self.data[self.conf > conf_thresh]
            if boxes.shape[0] == 0:
                return self._new()
            cls_score = boxes[..., self._cls_idx:].max(-1)
            conf = boxes[..., 4] * cls_score
        else:
            boxes = self.data.copy()
            conf = boxes[..., self._cls_idx:].max(-1)
        mask = conf >= conf_thresh
        boxes = boxes[mask]
        if boxes.shape[0] == 0:
            return self._new()
        shape = (*boxes.shape[:-1], 1)
        return self.__class__(self._concat(
            (boxes[:, :4], conf[mask].reshape(shape), np.argmax(boxes[..., self._cls_idx:], axis=-1).reshape(shape)),
            axis=-1), is_xywh=self.is_xywh)

    def clip(self, w, h=None):
        if h is None:
            self.data[..., :4] = np.clip(self.data[..., :4], a_min=0, a_max=w)
        else:
            self.data[..., [0, 2]] = np.clip(self.data[..., [0, 2]], a_min=0, a_max=w)
            self.data[..., [1, 3]] = np.clip(self.data[..., [1, 3]], a_min=0, a_max=h)

    def nms(self, nms_thresh):
        if self.dim != 2:
            raise NotImplementedError("Dim must be 2")
        if self.conf is None:
            raise IndexError("boxes to nms must have confidences")
        boxes = self.xyxy
        ranking_idxs = np.argsort(self.conf, axis=0)[::-1]
        keep = []
        while ranking_idxs.size > 0:
            i = int(ranking_idxs[0])
            keep.append(i)
            if ranking_idxs.shape[0] == 1:
                break
            iou = self.iou_mat(self.__class__(boxes[i].reshape((1, 4))),
                               self.__class__(boxes[ranking_idxs[1:]])).reshape(-1)
            ranking_idxs = ranking_idxs[1:][iou <= nms_thresh]
        return self[np.array(keep)]

    @staticmethod
    def iou(b1, b2, eps=1e-7):
        boxes1, boxes2 = b1.xyxy, b2.xyxy

        inter_upperlefts = np.maximum(boxes1[:, :2], boxes2[:, :2])
        inter_lowerrights = np.minimum(boxes1[:, 2:], boxes2[:, 2:])
        inters = inter_lowerrights - inter_upperlefts
        inters[inters < 0] = 0

        inter_areas = inters[:, 0] * inters[:, 1]
        union_areas = b1.box_area + b2.box_area - inter_areas
        return inter_areas / (union_areas + eps)

    @staticmethod
    def iou_mat(b1, b2, eps=1e-8):
        if not b1.dim == b2.dim == 2:
            raise NotImplementedError("iou_mat only for boxes dim=2")
        boxes1, boxes2 = b1.xyxy, b2.xyxy

        inter_upperlefts = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
        inter_lowerrights = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])

        inters = (inter_lowerrights - inter_upperlefts)
        inters[inters < 0.0] = 0.0

        inter_areas = inters[:, :, 0] * inters[:, :, 1]
        union_areas = b1.box_area[:, None] + b2.box_area - inter_areas
        return inter_areas / (union_areas + eps)

    @staticmethod
    def giou(boxes1, boxes2, eps=1e-8):
        b1, b2 = boxes1.xyxy, boxes2.xyxy
        intersect_mins = torch.max(b1[..., :2], b2[..., :2])
        intersect_maxes = torch.min(b1[..., 2:], b2[..., 2:])
        intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        union_area = boxes1.box_area + boxes2.box_area - intersect_area
        iou = intersect_area / (union_area + eps)

        enclose_mins = torch.min(b1[..., :2], b2[..., :2])
        enclose_maxes = torch.max(b1[..., 2:], b2[..., 2:])
        enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))

        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
        return iou - (enclose_area - union_area) / (enclose_area + eps)


def squeeze_with_indices(xs: List[Union[Boxes, NumpyBoxes]]):
    indices = []
    for i, x in enumerate(xs):
        indices += ([i] * len(x))
    indices = torch.tensor(indices, device=x.device).view(-1, 1)
    return x.concat(xs, 0).add_attrs(indices)


if __name__ == '__main__':
    print(all_equal([1, 1, 1, 2]))
