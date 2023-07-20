from copy import deepcopy

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import math

import time
from typing import Optional, Union
import os
import shutil

from . import LOGGER, Logger, Timer, check_dir


class GradAccumulator:
    def __init__(self, amp=False):
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp)
        self.last_step_batches = -1

    def step(self, model, optimizer, loss, trained_batches, nbs=64, grad_max_norm=10.0):
        self.scaler.scale(loss).backward()
        if trained_batches - self.last_step_batches < nbs:
            return False

        self.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_max_norm)
        self.scaler.step(optimizer)
        self.scaler.update()
        optimizer.zero_grad()
        self.last_step_batches = trained_batches
        return True


class WarmupMonitor:
    def __init__(self, warmup_batches, warmup_bias_lr, warmup_momentum, nbs):
        self.warmup_batches = max(warmup_batches, 100)
        self.warmup_bias_lr = warmup_bias_lr
        self.warmup_momentum = warmup_momentum
        self.nbs = nbs
        self.trained_batches = 0

    def _interpolate(self, y):
        return np.interp(self.trained_batches, [0, self.warmup_batches], y)

    def _warmup(self, optimizer, lr_now, momentum_now, batch_size):
        if self.trained_batches <= self.warmup_batches:
            accumulate = max(1, self._interpolate([1, self.nbs / batch_size]).round())
            for j, x in enumerate(optimizer.param_groups):
                # bias if j==0
                x['lr'] = self._interpolate(
                    [self.warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * lr_now])  # lr_now=lf(epoch)
                if 'momentum' in x:
                    x['momentum'] = self._interpolate([self.warmup_momentum, momentum_now])
            return accumulate

    def step(self, optimizer, lr_now, momentum_now, batch_size):
        # warmup_batches = max(round(self.warmup_epochs * num_batches), 100)
        accumulate = self._warmup(optimizer, lr_now, momentum_now, batch_size)
        self.trained_batches += batch_size
        return accumulate


class Scheduler:
    def __init__(self, optimizer, lrf, num_epochs, cos_lr=True):
        self.lrf = lrf

        if cos_lr:
            y1 = 1.0
            self.lf = lambda x: ((1 - math.cos(x * math.pi / num_epochs)) / 2) * (lrf - y1) + y1
        else:
            self.lf = lambda x: (1 - x / num_epochs) * (1.0 - lrf) + lrf
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self.lf)

    def step(self):
        self.scheduler.step()

    def resume(self, epoch):
        self.scheduler.last_epoch = epoch - 1

    @property
    def lr(self):
        return self.scheduler.get_last_lr()

    def get_lr_now(self, epoch):
        return self.lf(epoch)


class ModelEMA:
    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        self.ema = deepcopy(model).eval()
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        self.updates += 1
        d = self.decay(self.updates)

        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v *= d
                v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        self.copy_attr(self.ema, model, include, exclude)

    @staticmethod
    def copy_attr(a, b, include=(), exclude=()):
        # Copy attributes from b to a, options to only include [...] and to exclude [...]
        for k, v in b.__dict__.items():
            if (len(include) and k not in include) or k.startswith('_') or k in exclude:
                continue
            else:
                setattr(a, k, v)


class YoloOptimizer:
    def __init__(self, model, cfg, num_batches):
        self.momentum = cfg["momentum"]
        self.nbs = cfg["nbs"]
        self.epochs = cfg["epochs"]
        self.optimizer = self.smart_optimizer(model, cfg["optimizer"], cfg["lr0"], self.momentum, cfg["weight_decay"])
        self.grad_accumulator = GradAccumulator(cfg["amp"])
        self.warmup_monitor = WarmupMonitor(round(num_batches * cfg["warmup_epochs"]), cfg["warmup_bias_lr"],
                                            cfg["warmup_momentum"], self.nbs)
        self.scheduler = Scheduler(self.optimizer, cfg["lrf"], self.epochs, cfg["cos_lr"])
        if cfg["ema"]:
            self.ema = ModelEMA(model)
        else:
            self.ema = None

        self.epoch = 0

    def check_resume(self, checkpoint):
        if "optimizer" in checkpoint and "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
            if start_epoch < 0 or start_epoch > self.epochs:
                LOGGER.Error(
                    f"Cannot resume YoloOptimizer, the epoch must be less than epochs: {self.epochs} and greater than 0.")
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.resume(start_epoch)
            self.epoch = start_epoch
            LOGGER.info(f"Resuming YoloOptimizer with epoch={start_epoch}")

    def optimize(self, loss_iter, model, batch_size):
        for loss in loss_iter:
            self.grad_accumulator.step(model, self.optimizer, loss, self.trained_batches, self.nbs)
            if self.ema is not None:
                self.ema.update(model)

            self.optimizer.zero_grad()
            lr_now = self.scheduler.get_lr_now(self.epoch)
            self.warmup_monitor.step(self.optimizer, lr_now, self.momentum, batch_size)
        self.scheduler.step()
        self.epoch += 1

    @property
    def trained_batches(self):
        return self.warmup_monitor.trained_batches

    @property
    def lrs(self):
        lrs = []
        for param_group in self.optimizer.param_groups:
            lrs.append(param_group["lr"])
        return lrs

    @staticmethod
    def sort_model_params(model):
        bias, weight_nodeacy, weight_decay = [], [], []
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)
        for v in model.modules():
            for p_name, p in v.named_parameters(recurse=0):
                if p_name == 'bias':  # bias (no decay)
                    bias.append(p)
                elif p_name == 'weight' and isinstance(v, bn):  # weight (no decay)
                    weight_nodeacy.append(p)
                else:
                    weight_decay.append(p)  # weight (with decay)
        return bias, weight_nodeacy, weight_decay

    @classmethod
    def smart_optimizer(cls, model, name, lr, momentum, decay):
        bias, weight_nodeacy, weight_decay = cls.sort_model_params(model)

        if name == 'Adam':
            if decay != 0.0:
                LOGGER.warning(
                    f"Adam with weight_decay={decay} may cause problems, you 'd better set the momentum=0.0")
            optimizer = torch.optim.Adam(bias, lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum
        elif name == 'AdamW':
            optimizer = torch.optim.AdamW(bias, lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == 'RMSProp':
            optimizer = torch.optim.RMSprop(bias, lr=lr, momentum=momentum)
        elif name == 'SGD':
            optimizer = torch.optim.SGD(bias, lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(f'Optimizer {name} not implemented.')

        optimizer.add_param_group({'params': weight_decay, 'weight_decay': decay})
        optimizer.add_param_group({'params': weight_nodeacy, 'weight_decay': 0.0})
        LOGGER.info(f"optimizer: {name}(lr={lr},"
                    f" param_groups {len(weight_nodeacy)} weights(decay=0.0), {len(weight_decay)} weight(decay={decay}), {len(bias)} bias)")
        return optimizer
