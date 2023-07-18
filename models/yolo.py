import torch.nn.functional as F
import torch
from torch import nn
from copy import deepcopy
from pathlib import Path
import sys

sys.path.append("..")
from models.experimental import *
from utils import LOGGER


def make_divisible(x, divisor):
    # Returns x evenly divisble by divisor
    return math.ceil(x / divisor) * divisor


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True


def fuse_conv_and_bn(conv, bn):
    # Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          dilation=conv.dilation,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # Prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # Prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


class Detect1(nn.Module):
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else torch.cat(z, 1)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class Detect(nn.Module):

    def __init__(self, nc=80, anchors=()):  # detection layer
        super().__init__()
        self.stride = None  # strides computed during build
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)
        self.register_buffer('anchor_grid', a.clone())

    def forward(self, x):
        for layer_idx in range(self.nl):
            # x[layer_idx] = global_feature(x[layer_idx])
            x[layer_idx] = self.m[layer_idx](x[layer_idx])
            x[layer_idx] = self._decode(x[layer_idx], layer_idx)
        return x if self.training else torch.cat(x, dim=1)

    def _decode3(self, x, layer_idx):
        bs, _, ny, nx = x.shape
        x = x.view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        return x

    def _decode(self, x, layer_idx):
        grid_size = x.size(2)
        if self.grid[layer_idx].shape[2:4] != (grid_size, grid_size):
            self.grid[layer_idx] = self.generate_grid(grid_size, device=x.device)

        stride = self.stride[layer_idx]
        anchors = self.anchor_grid[layer_idx].view(-1, 2)

        batch_size = x.size(0)
        x = x.view(batch_size, -1, grid_size * grid_size)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.no)
        x[:] = torch.sigmoid(x[:])

        x[:, :, :2] = 2.0 * x[:, :, :2] - 0.5  # (0,1) -> (-0.5,1.5)
        x[:, :, :2] = (x[:, :, :2] + self.grid[layer_idx]) * stride

        anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
        x[:, :, 2:4] = (2 * x[:, :, 2:4]) ** 2 * anchors  # [0,inf)*wh -> (0,4)*wh
        return x

    def _decode2(self, x, layer_idx):
        grid_size = x.size(2)
        if self.grid[layer_idx].shape[2:4] != (grid_size, grid_size):
            self.grid[layer_idx] = self.generate_grid(grid_size, device=x.device)
        anchors = self.anchor_grid[layer_idx].view(-1, 2)

        batch_size = x.size(0)
        x = x.view(batch_size, -1, grid_size * grid_size)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.no)
        x[:] = torch.sigmoid(x[:])

        anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
        x[:, :, 2:4] = (2 * x[:, :, 2:4]) ** 2 * anchors  # [0,inf)*wh -> (0,4)*wh
        x[:, :, :2] = ((x[:, :, :2] * 2.0 - 1.0) * x[:, :, 2:4].clone() * 0.5 + self.grid[layer_idx]) * self.stride[
            layer_idx]
        return x

    def generate_grid(self, grid_size, device=None):
        grid_arange = torch.arange(grid_size, device=device)
        y_offset, x_offset = torch.meshgrid(grid_arange, grid_arange)
        x_offset = x_offset.reshape(-1, 1)
        y_offset = y_offset.reshape(-1, 1)
        grid = torch.cat((x_offset, y_offset), dim=1).repeat(1, self.na).view(1, -1, 2)
        return grid


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict

        # Define model
        if nc and nc != self.yaml['nc']:
            print('Overriding %s nc=%g with nc=%g' % (cfg, self.yaml['nc'], nc))
            self.yaml['nc'] = nc  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist, ch_out

        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 128  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        initialize_weights(self)
        self.info()
        print('')

    def forward(self, x):
        y = []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            # print([a.shape for a in x])
            x = m(x)
            y.append(x if m.i in self.save else None)  # save output
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for f, s in zip(m.f, m.stride):  #  from #m.f :feature idx
            mi = self.model[f % m.i]
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for f in sorted([x % m.i for x in m.f]):  #  from
            b = self.model[f].bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%g Conv2d.bias:' + '%10.3g' * 6) % (f, *b[:5].mean(1).tolist(), b[5:].mean()))

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ', end='')
        for m in self.model.modules():
            if type(m) is Conv:
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                m.bn = None  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self


class BaseModel(nn.Module):
    def forward(self, x):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        return x

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


class ClassificationModel(BaseModel):
    # YOLOv5 classification model
    def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):  # yaml, model, number of classes, cutoff index
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        # Create a YOLOv5 classification model from a YOLOv5 detection model
        # if isinstance(model, DetectMultiBackend):
        #     model = model.model  # unwrap DetectMultiBackend
        model.model = model.model[:cutoff]  # backbone
        m = model.model[-1]  # last layer
        ch = m.conv.in_channels if hasattr(m, 'conv') else m.cv1.conv.in_channels  # ch into module
        c = Classify(ch, nc)  # Classify()
        c.i, c.f, c.type = m.i, m.f, 'models.common.Classify'  # index, from, type
        model.model[-1] = c  # replace
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg):
        self.model = None


def parse_model(d, ch):  # model_dict, input_channels(3)
    print('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2)  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [nn.Conv2d, Conv, Bottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP, C3]:
            c1, c2 = ch[f], args[0]

            # Normal
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1.75  # exponential (default 2.0)
            #     e = math.log(c2 / ch[1]) / math.log(2)
            #     c2 = int(ch[1] * ex ** e)
            # if m != Focus:
            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2

            # Experimental
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1 + gw  # exponential (default 2.0)
            #     ch1 = 32  # ch[1]
            #     e = math.log(c2 / ch1) / math.log(2)  # level 1-n
            #     c2 = int(ch1 * ex ** e)
            # if m != Focus:
            #     c2 = make_divisible(c2, 8) if c2 != no else c2

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
        elif m is Detect:
            f = f or list(reversed([(-1 if j == i else j - 1) for j, x in enumerate(ch) if x == no]))
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        print('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


def make_detect_weights(detect, num_classes):
    if detect.nc == num_classes:
        return detect

    def cls_part(x, na, num_attr_old, num_attr_new):
        parts = []
        for i in range(na):
            idx = i * num_attr_old
            part = x[idx: idx + num_attr_new]
            parts.append(part)
        return torch.cat(parts, dim=0)

    no_old = detect.no
    detect.no = detect.nc + 5
    channels = detect.no * detect.na
    if detect.nc > num_classes:
        detect.nc = num_classes
        for i in range(len(detect.m)):
            detect.m[i].out_channels = channels

            weights = detect.m[i].weight.data
            bias = detect.m[i].bias.data
            new_weights = cls_part(weights, detect.na, no_old, detect.no)
            new_bias = cls_part(bias, detect.na, no_old, detect.no)
            detect.m[i].weight = nn.Parameter(new_weights)
            detect.m[i].bias = nn.Parameter(new_bias)
        LOGGER.warning(f"Num_classes of the model(={no_old - 5}) is greater than num_classes in need(={num_classes}).\n"
                       "Removing some weights from model.model[-1].\n")
        return detect
    detect.nc = num_classes
    for i in range(len(detect.m)):
        detect.m[i].out_channels = channels

        weights = detect.m[i].weight.data
        bias = detect.m[i].bias.data
        shape = weights.shape

        new_weights = torch.cat((weights, torch.normal(0, 0.01, (channels - shape[0], shape[1], shape[2], shape[3]))),
                                dim=0)
        new_bias = torch.cat((bias, torch.normal(0, 0.01, (channels - shape[0]))), dim=0)
        detect.m[i].weight = nn.Parameter(new_weights)
        detect.m[i].bias = nn.Parameter(new_bias)
    LOGGER.warning(f"Num_classes of the model(={no_old - 5}) is less than num_classes in need(={num_classes}).\n"
                   "Padding weights on model.model[-1].\n")
    return detect


def yolov5(model="s", num_classes=None, requires_ckpt=False):
    if model in ["n", "s", "m", "l", "x"]:
        model = f"models/yolov5{model}.pt"
        # return Model(f"models/yolov5{model}.yaml", nc=num_classes)
    ckpt = torch.load(model, map_location="cpu")
    model = ckpt.pop("model")

    detect = model.model[-1]
    if type(detect) == Detect and num_classes is not None:
        model.model[-1] = make_detect_weights(detect, num_classes)
    if requires_ckpt:
        return model, ckpt
    return model
