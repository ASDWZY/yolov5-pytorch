import json
import os
import threading
import subprocess
import sys
from typing import Union, Tuple, Optional

import cv2
import numpy as np
from matplotlib import pyplot as plt

from . import LOGGER


def clip(x, xmin, xmax):
    return min(max(x, xmin), xmax)


class Image:
    def __init__(self, img_data: np.ndarray, rgb=True):
        if type(img_data) != np.ndarray:
            LOGGER.Error("data_type of class Image must be np.ndarray", "TypeError")
        self.data = img_data
        self.rgb = rgb

    def __str__(self):
        string = "Image("
        string += f"RGB={self.rgb}, w={self.width}, h={self.height}, c={self.channels}, dtype={self.dtype})"
        return string

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def width(self):
        return self.shape[1]

    @property
    def height(self):
        return self.shape[0]

    @property
    def channels(self):
        return self.shape[2]

    @property
    def rgb_data(self):
        if self.rgb:
            return self.data
        return self.data[:, :, ::-1].astype(np.uint8)

    @property
    def bgr_data(self):
        if self.rgb:
            return self.data[:, :, ::-1].astype(np.uint8)
        return self.data

    @classmethod
    def read(cls, path: str, gray=False):
        if not os.path.isfile(path):
            return "not exists"
        if gray:
            data = cv2.imread(path, 0)
            data = data.view(data.shape[0], data.shape[1], 1)
        else:
            data = cv2.imread(path)
        if data is None:
            return "not img"
        return Image(data, False)

    def copy(self):
        return Image(self.data.copy(), self.rgb)

    def save(self, path):
        cv2.imwrite(path, self.bgr_data)

    def show(self, winname="img", wait_key=0):
        cv2.imshow(winname, self.bgr_data)
        if wait_key is not None:
            cv2.waitKey(wait_key)

    def plot(self, show=True):
        fig = plt.imshow(self.rgb_data)
        if show:
            plt.show()
        return fig

    def to_bgr(self):
        if self.rgb:
            return Image(self.data[:, :, ::-1].astype(np.uint8), False)
        return self

    def to_rgb(self):
        if self.rgb:
            return self
        return Image(self.data[:, :, ::-1].astype(np.uint8), True)

    def inverse(self):
        return Image(self.data[:, :, ::-1].astype(np.uint8), not self.rgb)

    def resize(self, size=(0, 0), fx=1, fy=1):
        new_img = cv2.resize(self.data, size, fx=fx, fy=fy)
        return Image(new_img, self.rgb)

    def cut(self, x1, y1, x2, y2):
        return Image(self.data[int(max(y1, 0)):int(min(y2, self.height)), int(max(x1, 0)):int(min(x2, self.width))],
                     self.rgb)

    def letterbox(self, w, h=None, border=None):
        if h is None:
            h = w
        factor = min(w / self.width, h / self.height)
        resized_img = cv2.resize(self.data, (0, 0), fx=factor, fy=factor)
        img_h, img_w, channels = resized_img.shape

        if border is not None:
            new_img = np.zeros((h, w, channels), dtype=np.uint8)
            new_img[:] = border
        else:
            new_img = np.random.randint(0, 255, (h, w, channels), dtype=np.uint8)

        y_offset = (h - img_h) // 2
        x_offset = (w - img_w) // 2

        new_img[y_offset:y_offset + img_h, x_offset:x_offset + img_w] = resized_img
        return Image(new_img, self.rgb)


class FfmpegCommand:
    def __init__(self, ffmpeg="ffmpeg"):
        self._command = [ffmpeg]
        self._popen = None

    def add(self, *args, **kwargs):
        for arg in args:
            self._command.append(str(arg))
        for k, v in kwargs.items():
            self._command.append(f"-{k}")
            self._command.append(str(v))

    def run(self, check=False):
        self._popen = subprocess.Popen(self._command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if sys.platform == "win32":
            self._popen.stderr.close()
        if check:
            out, err = self._popen.communicate()
            if self._popen.returncode != 0:
                raise Exception('ffprobe', out, err)
            return out

    def release(self):
        if self._popen:
            self._popen.stdout.close()
            if sys.platform != "win32":
                self._popen.stderr.close()
            self._popen.kill()
            self._popen.wait()

    def read(self, bufsize):
        return self._popen.stdout.read(bufsize)


class Video:
    def __init__(self, src: str, fps=None, img_shape=None, is_rgb=False, ffmpeg="ffmpeg", timeout=5):
        self.command = FfmpegCommand(ffmpeg)
        if timeout is not None:
            self.command.add(timeout=int(1000 * 1000 * timeout))
        if src.startswith("rtsp://"):
            self.command.add(rtsp_transport="tcp")
        self.command.add(i=src, f="rawvideo", pix_fmt="rgb24" if is_rgb else "bgr24")

        if fps is not None:
            self.command.add(r=fps)
        if img_shape is None:
            probe = self.probe_video(src)
            cap_info = next(x for x in probe["streams"] if x["codec_type"] == "video")
            self.width = cap_info["width"]
            self.height = cap_info["height"]
        else:
            self.height, self.width = img_shape
            self.command.add(s=str(self.width) + '*' + str(self.height))
        self.command.add("pipe:1")
        self.command.run()

    def read(self):
        buffer = self.command.read(self.width * self.height * 3)
        if buffer:
            frame = np.frombuffer(buffer, dtype=np.uint8).reshape((self.height, self.width, 3))
            return True, frame
        return False, None

    def __del__(self):
        self.command.release()

    @classmethod
    def probe_video(cls, src):

        p = subprocess.Popen(['ffprobe', '-show_format', '-show_streams', '-of', 'json', src], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        out, err = p.communicate()
        if p.returncode != 0:
            raise Exception('ffprobe', out, err)
        return json.loads(out.decode('utf-8'))


class VideoCapture:
    def __init__(self, src, ffmpeg="ffmpeg"):
        if src.startswith("rtsp://"):
            self.video = Video(src, ffmpeg=ffmpeg)
        else:
            self.video = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.display_width)

    def read(self):
        ret, frame = self.video.read()
        if ret:
            return Image(frame, rgb=False)
        else:
            return None

    def __iter__(self):
        while True:
            ret, frame = self.video.read()
            if ret:
                yield frame

    def show(self, winname="frame"):
        for img in self:
            if img is not None:
                img.show(winname, wait_key=1)


if __name__ == '__main__':
    video = VideoCapture("rtsp://admin:123456@10.64.45.202:554/h264/ch1/main/av_stream")
