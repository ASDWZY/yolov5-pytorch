import re

from shutil import rmtree
import time

import os

import inspect

import yaml

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Timer:
    clock_names = ["days", "hours", "minutes", "seconds"]
    clock_strides = strides = [24, 60, 60]

    def __init__(self, interval=None):
        self.start_time = time.time()

        self.end_time = None if interval is None else self.start_time + interval

    def reset(self):
        self.start_time = time.time()

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()

    @property
    def interval(self):
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    @classmethod
    def get_interval_segments(cls, interval):
        segments = {}
        temp = interval
        stride_m = 1
        for stride in cls.clock_strides:
            stride_m *= stride

        for name, stride in zip(cls.clock_names, cls.clock_strides):
            a, temp = divmod(temp, int(stride_m))
            segments[name] = int(a)
            stride_m /= stride
        segments[cls.clock_names[-1]] = int(temp)
        return segments

    def get_interval_formats(self, formats=None, hide0=True):
        if formats is None:
            formats = ["{days}days", "{hours}hours", "{minutes}minutes", "{seconds}seconds"]
        new_formats = []
        segments = self.get_interval_segments(self.interval)
        for name in self.clock_names:
            value = segments[name]
            if not (value == 0 and hide0 and name!="seconds"):
                for fmt in formats:
                    if name in fmt:
                        new_formats.append(fmt.replace("{"+name+"}", "{}").format(value))
        return ", ".join(new_formats)


def check_dir(dir, clear=False):
    if not os.path.exists(dir):
        os.makedirs(dir)
    elif clear:
        rmtree(dir)
        os.makedirs(dir)


def check_packages(packages):
    requirements = {}
    for p in packages:
        try:
            __import__(p)
            requirements[p] = True
        except ModuleNotFoundError:
            requirements[p] = False
    return requirements


class StreamLogger:
    level_dict = {
        "blue": 44,
        "purple": 45,
        "info": 42,
        "warning": 43,
        "error": 41,
        "step": 46,
    }

    def __init__(self, mode=0, color=30):
        self.mode = mode
        self.color = color

    def log(self, message, level, end="\n"):
        color_level = self.level_dict[level]
        self.colorful_print(message, self.mode, self.color, color_level, end)

    @staticmethod
    def colorful_print(item, mode=0, color_foreground=37, color_background=40, end="\n"):
        print(f"\033[{mode};{color_foreground};{color_background}m{item}\033[0m", end=end)


class FileLogger:
    def __init__(self, filename):
        self.filename = filename

    def log(self, message, level, end="\n"):
        with open(self.filename, "a") as log_file:
            log_file.write(message + end)


class Logger:
    def __init__(self, name):
        self.name = name
        self.loggers = []
        self.level_enabled = {
            "info": True,
            "error": True,
            "warning": True,
            "blue": True,
            "purple": True,
            "step": True,
        }

    def add(self, filename=None, formats="{name} - {level}: {message}"):
        formats = self.formats_log(formats, ["name", "level", "message"])
        if filename:
            self.loggers.append((FileLogger(filename), formats))
        else:
            self.loggers.append((StreamLogger(), formats))

    def log(self, messages, level, end="\n"):
        if not self.level_enabled[level]:
            return "disabled"
        message = "\t".join(messages)
        for logger, formats in self.loggers:
            msg = formats.format(self.name, level.upper(), message)
            logger.log(msg, level, end)
        return "enabled"

    def __getattr__(self, level):
        def log(*message, end="\n"):
            self.log(message, level, end)

        return log

    def disable(self, log_name):
        self.level_enabled[log_name] = False

    def enable(self, log_name):
        self.level_enabled[log_name] = True

    def Error(self, messages, error_type="NotImplementedError"):
        def get_line_content(line_number, filename):
            with open(filename, 'r') as file:
                lines = file.readlines()
            if line_number <= len(lines):
                line_content = lines[line_number - 1].strip()
                return line_content
            else:
                return None

        frame = inspect.currentframe().f_back
        frames = []
        while frame:
            frames.append(frame)
            frame = frame.f_back
        frames.reverse()
        for frame in frames[:-1]:
            for logger, formats in self.loggers:
                line_number, filename = frame.f_lineno, inspect.getframeinfo(frame).filename
                logger.log(
                    f"in {filename}, line{line_number}, function {frame.f_code.co_name} :\n{get_line_content(line_number, filename)}",
                    "error", "\n")
        msg = f"{error_type} in {inspect.getframeinfo(frames[-1]).filename}, line{frames[-1].f_lineno}, function {frames[-1].f_code.co_name} : {messages}"
        for logger, formats in self.loggers:
            logger.log(msg, "error", "\n")
        exit(-1)

    @staticmethod
    def formats_log(formats, names):
        for i, name in enumerate(names):
            formats = formats.replace("{" + name + "}", "{" + str(i) + "}")
        return formats


LOGGER = Logger("yolov5")
LOGGER.add()


class ParamsList(dict):
    def __init__(self, *args, **kwargs):
        if args:
            super().__init__(*args)
        else:
            super().__init__(**kwargs)
        self.key_name = "keys"
        self.value_name = "values"
        self.width_offset = 2

    def set_print_format(self, key_name="keys", value_name="values", width_offset=2):
        self.key_name = key_name
        self.value_name = value_name
        self.width_offset = width_offset

    def print_line(self, string, key, value, key_length, value_length):
        key = key.center(key_length, " ")
        value = value.center(value_length, " ")
        c = chr(0x2502)
        string += c + str(key) + c + str(value) + c + "\n"
        return string

    def __str__(self):
        string = "\n"
        key_lengths = [len(self.key_name)]
        value_lengths = [len(self.value_name)]
        items = [(key, value) for key, value in self.items() if
                 value not in [self.key_name, self.value_name, self.width_offset]]
        for key, value in items:
            key_lengths.append(len(str(key)))
            value_lengths.append(len(str(value)))

        key_length = max(key_lengths) + self.width_offset
        value_length = max(value_lengths) + self.width_offset
        string += chr(0x250c) + chr(0x2500) * key_length + chr(0x252c) + chr(0x2500) * value_length + chr(0x2510) + "\n"
        string = self.print_line(string, self.key_name, self.value_name, key_length, value_length)
        string += chr(0x255e) + chr(0x2550) * key_length + chr(0x256a) + chr(0x2550) * value_length + chr(0x2561) + "\n"
        for key, value in items:
            string = self.print_line(string, f"{key}", f"{value}", key_length, value_length)
        string += chr(0x2514) + chr(0x2500) * key_length + chr(0x2534) + chr(0x2500) * value_length + chr(0x2518) + "\n"
        return string

    def __repr__(self):
        return str(self)

    def __getattr__(self, item):
        return self[item]

    def __setattr__(self, key, value):
        self[key] = value


def load_yaml(yaml_file):
    with open(yaml_file, errors='ignore', encoding='utf-8') as f:
        s = f.read()
        if not s.isprintable():
            s = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+', '', s)
        return yaml.safe_load(s)




if __name__ == '__main__':
    LOGGER.Error("hello")
    print(90)
