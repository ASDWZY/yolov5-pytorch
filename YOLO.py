import cv2
import numpy as np
from abc import ABC, abstractmethod
from utils import LOGGER, check_packages, Timer, load_yaml
from utils.image import Image
from utils.box import NumpyBoxes, squeeze_with_indices
from typing import Tuple, Union, List, Iterator
from functools import wraps

requirements = check_packages(["torch", "onnxruntime", "tensorrt", "matplotlib"])

if requirements["torch"]:
    import torch
    from torch import nn
    from models.yolo import DetectionModel
    from utils.box import Boxes
else:
    LOGGER.warning("Please install torch.")
if requirements["matplotlib"]:
    import matplotlib.pyplot as plt

ImagesTyping = Union[Tuple[Image], List[Image]]


class ImageResult:
    def __init__(self, boxes: Union[Boxes, NumpyBoxes], img: Image, names: List[str]):
        self.boxes = boxes
        self.origin_img = img
        self.names = names

    def __str__(self):
        return f"YoloOut(origin_img={self.origin_img},boxes={self.boxes})"

    def __repr__(self):
        return str(self)

    def draw(self, colors=None, label_color=(255, 255, 255), thickness: int = 2, font=cv2.FONT_HERSHEY_SIMPLEX,
             font_scale=1.0):
        if colors is None:
            colors = self.rainbow_colors(len(self.names))
        img = self.origin_img.rgb_data.copy()
        self.boxes.draw(img, self.names, colors, label_color, thickness, font, font_scale)
        return Image(img, True)

    def plot(self, colors=None, label_color="w", thickness: int = 2, show=True):
        fig = plt.imshow(self.origin_img.rgb_data)
        if colors is None:
            colors = ["g"] * len(self.names)
        self.boxes.plot(fig.axes, self.names, colors, label_color, thickness)
        if show:
            plt.show()
        return fig

    @staticmethod
    def rainbow_colors(count, uint8=True):
        def rainbow_interpolate(x):
            # 1 red to yellow
            if x < 0.2:
                return 1, x / 0.2, 0
            # 2 yellow to green
            if x < 0.4:
                return 1.0 - (x - 0.2 * 1) / 0.2, 1, 0
            # 3 green to cray
            if x < 0.6:
                return 0, 1, (x - 0.2 * 2) / 0.2
            # 4 cray to blue
            if x < 0.8:
                return 0, 1.0 - (x - 0.2 * 3) / 0.2, 1
            # 5 blue to purple
            return (x - 0.2 * 5) / 0.2, 0, 1

        colors = []
        for i in np.arange(0, 1, 1 / count):
            color = rainbow_interpolate(i)
            if uint8:
                color = tuple([int(c * 255) for c in color])
            colors.append(color)
        return colors


class VideoResult:
    def __init__(self, results_generator: Iterator[ImageResult], fps=None, total_frames=None):
        self.results_iter = results_generator
        self.result0 = next(iter(self.results_iter))
        self.fps = fps
        self.total_frames = total_frames
        self.frame_id = 0
        self.timer = Timer()

    def __iter__(self):
        self.timer.reset()
        if self.frame_id == 0:
            yield self.result0
        for out in self.results_iter:
            self.frame_id += 1
            yield out

    def __str__(self):
        string = f"YoloOutIter(result0={self.result0}, frame_id={self.frame_id}, fps={self.fps}, total_frames={self.total_frames})"
        return string

    def __repr__(self):
        return str(self)

    def _progress_format(self, seconds_timer=None):
        string = f"({self.timer.get_interval_formats()})"
        if seconds_timer is not None:
            string += f"/({seconds_timer.get_interval_formats()})"
        string += f", frame {self.frame_id}"
        if self.total_frames is not None:
            string += f"/{self.total_frames}"

        return string

    def write(self, save_path, colors=None, fourcc="mp4v", seconds=None, fps=None):
        fourcc = cv2.VideoWriter_fourcc(*fourcc)
        if fps is None:
            fps = self.fps if self.fps else 25
        writer = cv2.VideoWriter(save_path, fourcc, fps, (self.img_shape[1], self.img_shape[0]),
                                 True)
        LOGGER.info(f"Writing detection results in {save_path}")
        names = self.names
        if colors is None:
            colors = ImageResult.rainbow_colors(len(names))

        timer = Timer(seconds)
        for out in self:
            print(f"\rWrote for {self._progress_format(timer)}, ", end="")
            if self.timer.interval >= seconds:
                break
            if out is None:
                continue
            frame = out.draw(names, colors)
            writer.write(frame.bgr_data)

        writer.release()

    def display(self, names, colors=None, winname="frame"):
        LOGGER.info(f"Displaying detection results")
        if colors is None:
            colors = ImageResult.rainbow_colors(len(names))
        for out in self:
            if out is None:
                continue
            frame = out.draw(names, colors)
            frame.show(winname, wait_key=1)

    @property
    def img_shape(self):
        return self.result0.origin_img.shape

    @property
    def names(self):
        return self.result0.names


def timer_method(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with Timer() as timer:
            temp = func(self, *args, **kwargs)
        getattr(self, "times").__setitem__(func.__name__, timer.interval)
        return temp

    return wrapper


class Model(ABC):
    boxes_class = NumpyBoxes

    def __init__(self, model_type="yolov5"):
        self.postprocess_func = {
            "yolov5": self.yolov5_nms,
            "yolov8": self.yolov8_nms
        }[model_type]
        self.times = {}

    @property
    def is_numpy(self):
        return self.boxes_class == NumpyBoxes

    @timer_method
    def preprocess(self, imgs: ImagesTyping, imgsz: int):
        x = []
        for img in imgs:
            x.append(img.letterbox(imgsz).rgb_data)
        x = np.stack(x, axis=0).transpose((0, 3, 1, 2)).astype(np.float32)
        x /= 255.0
        return x

    @timer_method
    @abstractmethod
    def inference(self, x):
        pass

    @timer_method
    def postprocess(self, pred, imgsz=None, img_shapes=None, conf_thresh=0.5, nms_thresh=0.4):
        return self.postprocess_func(pred, imgsz, img_shapes, conf_thresh, nms_thresh)

    def __call__(self, imgs: ImagesTyping, imgsz: int, conf_thresh: float = 0.5, nms_thresh: float = 0.4):
        return self.postprocess(self.inference(self.preprocess(imgs, imgsz)), imgsz, [img.shape[:2] for img in imgs],
                                conf_thresh, nms_thresh)

    @classmethod
    def yolov5_nms(cls, prediction, imgsz=None, img_shapes=None, conf_thresh=0.5, nms_thresh=0.4):
        is_numpy = cls.boxes_class == NumpyBoxes
        outputs = []
        for batch_idx in range(prediction.shape[0]):
            boxes = cls.boxes_class(prediction[batch_idx], is_xywh=True).select_by_conf(conf_thresh)
            if len(boxes) == 0:
                outputs.append(cls.boxes_class())
                continue
            boxes.to_xyxy()
            if imgsz is not None and img_shapes is not None:
                boxes.reverse_letterbox(img_shapes[batch_idx], imgsz)
            output = []
            for pred_cls in boxes.unique_classes:
                output.append(boxes.get_cls_boxes(pred_cls).nms(nms_thresh).data)
            outputs.append(
                cls.boxes_class(np.concatenate(output, axis=0) if is_numpy else torch.cat(output, dim=0)))
        return outputs

    @classmethod
    def yolov8_nms(cls, prediction, imgsz=None, img_shapes=None, conf_thresh=0.5, nms_thresh=0.4):
        is_numpy = cls.boxes_class == NumpyBoxes
        prediction = prediction.transpose((0, 2, 1)) if is_numpy else prediction.permute(0, 2, 1)
        outputs = []
        for batch_idx in range(prediction.shape[0]):
            boxes = cls.boxes_class(prediction[batch_idx], is_xywh=True, has_conf=False).select_by_conf(conf_thresh)
            if len(boxes) == 0:
                outputs.append(cls.boxes_class())
                continue
            boxes.to_xyxy()
            if imgsz is not None and img_shapes is not None:
                boxes.reverse_letterbox(img_shapes[batch_idx], imgsz)
            output = []
            for pred_cls in boxes.unique_classes:
                output.append(boxes.get_cls_boxes(pred_cls).nms(nms_thresh).data)
            outputs.append(
                cls.boxes_class(np.concatenate(output, axis=0) if is_numpy else torch.cat(output, dim=0)))
        return outputs

    def warmup(self, imgsz, num_warmups):
        x = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)
        for i in range(num_warmups):
            self.inference(x)


class TorchModel(Model):
    boxes_class = Boxes
    if requirements["torch"]:
        devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())] + [torch.device("cpu")]

    def __init__(self, model, model_type="yolov5", device_id=None):
        super().__init__(model_type)
        if device_id is None:
            device_id = 0
        self.device = self.devices[device_id]
        if model.endswith(".yaml"):
            self.ckpt = {}
            self.model = DetectionModel(model)
        elif model.endswith(".pt"):
            self.ckpt = torch.load(model, map_location="cpu")
            self.model = self.ckpt.pop("model")
        else:
            LOGGER.Error("yolov5 model can only be created by yaml and pt file")
        self.model.eval().float().to(self.device)

    @staticmethod
    def _remove_weights(detect, no_new):
        def remove(x, na, num_attr_old, num_attr_new):
            news = []
            for i in range(na):
                idx = i * num_attr_old
                news.append(x[idx: idx + num_attr_new])
            return torch.cat(news, dim=0)

        no_old = detect.no
        detect.no = no_new
        for m in detect.m:
            m.out_channels = detect.no * detect.na

            weights = m.weight.data
            bias = m.bias.data
            m.weight = nn.Parameter(remove(weights, detect.na, no_old, detect.no))
            m.bias = nn.Parameter(remove(bias, detect.na, no_old, detect.no))
        return detect

    @staticmethod
    def _add_weights(detect, no_new):
        detect.no = no_new
        channels = detect.no * detect.na
        for i in range(len(detect.m)):
            detect.m[i].out_channels = channels

            weights = detect.m[i].weight.data
            bias = detect.m[i].bias.data
            shape = weights.shape

            new_weights = torch.cat(
                (weights, torch.normal(0, 0.01, (channels - shape[0], *shape[1:]), device=weights.device)),
                dim=0)
            new_bias = torch.cat((bias, torch.zeros((channels - shape[0]), device=weights.device)), dim=0)
            detect.m[i].weight = nn.Parameter(new_weights)
            detect.m[i].bias = nn.Parameter(new_bias)
        return detect

    def attempt_reset_nc(self, num_classes):
        detect = self.model.model[-1]
        if detect.nc == num_classes:
            return
        no_new = num_classes + 5
        old_nc = detect.nc
        detect.nc = num_classes
        if old_nc > num_classes:
            detect = self._remove_weights(detect, no_new)
        else:
            detect = self._add_weights(detect, no_new)
        self.model.model[-1] = detect
        LOGGER.warning(f"Reset num_classes from {old_nc} to {num_classes}, now model.model[-1]={detect}")

    @timer_method
    def preprocess(self, imgs: ImagesTyping, imgsz: int):
        x = []
        for img in imgs:
            x.append(torch.from_numpy(img.letterbox(imgsz).rgb_data).permute(2, 0, 1).to(self.device))
        x = torch.stack(x, dim=0).float()
        x /= 255.0
        return x

    @timer_method
    def inference(self, x):
        with torch.no_grad():
            return self.model(x)

    @timer_method
    def forward(self, x):
        return self.model(x)

    def warmup(self, imgsz, num_warmups):
        x = torch.rand((1, 3, imgsz, imgsz))
        for i in range(num_warmups):
            self.inference(x)

    def export_onnx(self, onnx_file, simplify):
        from export import export_onnx
        export_onnx(self.model, onnx_file, simplify)

    def train(self):
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True

    def eval(self):
        self.model.eval()

    def freeze(self):
        for param in self.model.model[:-1].parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.model.model[:-1].parameters():
            param.requires_grad = True

    @property
    def is_training(self):
        return self.model.training


if requirements["onnxruntime"]:
    import onnxruntime as ort
else:
    LOGGER.warning("Please install onnxruntime, use pip install onnxruntime-gpu.")


class OnnxModel(Model):
    if requirements["onnxruntime"]:
        devices = ort.get_available_providers()
    else:
        devices = []

    def __init__(self, model, model_type="yolov5", device_id=None):
        super().__init__(model_type)
        if device_id is None:
            device_id = 0
        self.device = self.devices[device_id]
        self.model = ort.InferenceSession(model, providers=[self.device])
        self.input_names = [inp.name for inp in self.model.get_inputs()]
        self.output_names = [out.name for out in self.model.get_outputs()]

    @timer_method
    def inference(self, *inputs):
        inputs_data = {name: x for name, x in zip(self.input_names, inputs)}
        return self.model.run(self.output_names, inputs_data)[0]


class YOLO:
    def __init__(self, model: str, names=None, device_id=None, yolo_type="yolov5", imgsz=640, conf_thresh=0.5,
                 nms_thresh=0.4):
        model_type = model.split(".")[-1]
        self.model_file = model
        try:
            self.model = {"pt": TorchModel, "onnx": OnnxModel}[model_type](model, yolo_type, device_id)

        except KeyError:
            LOGGER.Error(f"Unsupported model: {model}")

        self.names = names
        # if len(self.names) != self["nc"]:
        #     LOGGER.warning("nc")
        self.imgsz = imgsz
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

    def eval_mode(self):
        if self.is_torch:
            self.model.eval()

    def train_mode(self):
        if self.is_torch:
            self.model.train()

    def warmup(self, num_warmups=1):
        self.model.warmup(self.imgsz, num_warmups)

    def train(self, cfg: Union[str, dict], freeze_epochs=0, resume=False):
        if not self.is_torch:
            LOGGER.Error(f"Only TorchModel could train.")
        self.model.train()
        from train import Trainer
        if type(cfg) == str:
            cfg = load_yaml(cfg)
        trainer = Trainer(self, cfg)
        trainer.train(freeze_epochs, resume)

    def reset_num_classes(self, num_classes=None):
        if num_classes is None:
            num_classes = len(self.names)
        self.model.attempt_reset_nc(num_classes)

    def init_weights(self):
        if not self.is_torch:
            LOGGER.Error(f"Only TorchModel could init_weights.")
        self.model.model.init_weights()

    def preprocess(self, imgs: ImagesTyping):
        return self.model.preprocess([img.to_rgb() for img in imgs], self.imgsz)

    def inference(self, x):
        return self.model.inference(x)

    def forward(self, imgs: ImagesTyping):
        if not self.is_torch:
            LOGGER.Error("Only TorchModel could forward in training.")
        return self.model.forward(self.model.preprocess([img.to_rgb() for img in imgs], self.imgsz))

    def train_process(self, imgs: ImagesTyping, target_boxes: List[Boxes]):
        for x, img in zip(target_boxes, imgs):
            x.letterbox(img.shape[:2], self.imgsz)
            x.to(self.device)
        return self.forward(imgs), squeeze_with_indices(target_boxes)

    def postprocess(self, pred, img_shapes=None):
        return self.model.postprocess(pred, self.imgsz, img_shapes, self.conf_thresh, self.nms_thresh)

    def __call__(self, imgs: Union[Tuple[Image], List[Image], Image]) -> Union[List[ImageResult], ImageResult]:
        if type(imgs) == Image:
            pred = self.model([imgs], self.imgsz, self.conf_thresh, self.nms_thresh)
            return ImageResult(pred[0], imgs, self.names)
        pred = self.model(imgs, self.imgsz, self.conf_thresh, self.nms_thresh)
        return [ImageResult(x, img, self.names) for x, img in zip(pred, imgs)]

    def detect_video(self, src):
        video = cv2.VideoCapture(src)
        fps = int(video.get(cv2.CAP_PROP_FPS))
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        def detect_generator(video):
            if not video.isOpened():
                return []
            while True:
                ret, frame = video.read()
                if ret:
                    out = self(Image(frame, False))
                    yield out
                else:
                    return

        return VideoResult(detect_generator(video), fps, total_frames)

    def export_onnx(self, file=None, simplify=False):
        if not hasattr(self.model, "export_onnx"):
            LOGGER.Error(f"This model type: {self.model.__class__.__name__} could not export to onnx.")
        if file is None:
            file = self._replace_name("onnx")
        self.model.export_onnx(file, simplify)

    def _replace_name(self, file_type):
        model_type = self.model_file.split(".")[-1]
        return self.model_file.replace(model_type, file_type)

    @property
    def detect_times(self):
        return self.model.times

    @property
    def device(self):
        return self.model.device if hasattr(self.model, "device") else None

    @property
    def is_training(self):
        if type(self.model) == TorchModel:
            return self.model.is_training
        return False

    @property
    def is_torch(self):
        return type(self.model) == TorchModel

    def _getitem(self, item):
        if not self.is_torch:
            raise KeyError(str(item))
        return getattr(self.model.model.model[-1], item)

    def _setitem(self, key, value):
        if not self.is_torch:
            raise KeyError(str(key))
        return setattr(self.model.model.model[-1], key, value)

    def get_num_layers(self):
        return self._getitem("nl")

    def get_strides(self):
        return self._getitem("stride")

    def get_anchors(self):
        anchors = self._getitem("anchors")
        return anchors * self.get_strides().reshape(anchors.shape[0], 1, 1)

    def set_anchors(self, anchors):
        if type(anchors) != torch.tensor:
            anchors = torch.tensor(anchors, device=self.device)
        if len(anchors.shape) != 3:
            raise NotImplementedError("anchors must be 3D")
        if anchors.shape[0] != self.get_num_layers():
            raise NotImplementedError("Please reset num_layers first")
        if anchors.shape[2] != 2:
            raise NotImplementedError("anchors.shape[2] must be 2")
        self._setitem("anchors", anchors)
        self._setitem("na", anchors.shape[1])


if __name__ == '__main__':
    img = Image.read("imgs/0.jpg")
    model = YOLO("models/yolov5m.pt", names=["fire", "smoke"], device_id=0)
    # model.model.attempt_reset_nc(1)

    pred = model(img)
    print(pred)
    frame = pred.draw()
    frame.plot()

    # pred = model.detect_video("fires.mp4")
    # pred.write("result.mp4", seconds=20)
