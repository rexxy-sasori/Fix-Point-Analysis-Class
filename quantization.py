import copy

import functools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def quant_conv(model, n, m, args, offset=0, clip_val=None):
    bias = m.bias is not None
    init_args = {'weight_data': m.weight.data, 'bias_data': m.bias.data if bias else None}
    quant_args = {'quant_act': args.quant_act, 'num_bits_weight': args.num_bits_weight + offset,
                  'num_bits_act': args.num_bits_act}
    conv_args = {'in_channels': m.in_channels, 'out_channels': m.out_channels, 'kernel_size': m.kernel_size,
                 'stride': m.stride, 'padding': m.padding, 'groups': m.groups, 'bias': bias}
    conv = QConv2d(quant_scheme='quant', quant_args=quant_args, init_args=init_args, **conv_args)
    rsetattr(model, n, conv)
    print('CONV layer ' + n + ' quantized using ' + args.quant_scheme_conv)


def quant_linear(model, n, m, args, offset=0, clip_val=None):
    bias = False
    if m.bias is not None:
        bias = True
    quant_args = {'quant_act': args.quant_act, 'num_bits_weight': args.num_bits_weight + offset,
                  'num_bits_act': args.num_bits_act}
    fc_args = {'in_features': m.in_features, 'out_features': m.out_features, 'bias': bias}
    init_args = {'weight_data': m.weight.data, 'bias_data': m.bias.data if bias else None}
    lin = QLinear(quant_scheme=args.quant_scheme_fc, quant_args=quant_args, init_args=init_args, **fc_args)
    print('FC layer ' + n + ' quantized using ' + args.quant_scheme_fc)
    rsetattr(model, n, lin)


def quant_batch_norm(model, n, m, args):
    quant_args = {'num_bits_weight': args.num_bits_weight, 'num_bits_act': args.num_bits_act}
    bn_args = {'num_features': m.num_features, 'eps': m.eps, 'momentum': m.momentum, 'affine': m.affine,
               'track_running_stats': m.track_running_stats}
    init_args = {'weight_data': m.weight.data, 'bias_data': m.bias.data}
    bn = QBatchNorm2d(quant_args=quant_args, init_args=init_args, **bn_args)
    rsetattr(model, n, bn)
    print('BN layer ' + n + ' quantized')


def quant_relu(model, n, m, args, clip_val=None, offset=0, device='cpu'):
    relu_layer = QReLU(inplace=m.inplace, clip_val=clip_val, bits=args.num_bits_act + offset, device=device)
    rsetattr(model, n, relu_layer)


def quantize_model_uniform(model, args, a_range=()):
    a_range = copy.deepcopy(a_range.tolist())
    a_range.pop(0)

    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            quant_conv(model, n, m, args)

        if isinstance(m, nn.Linear):
            quant_linear(model, n, m, args)

        if isinstance(m, nn.BatchNorm2d) and args.quant_act:
            quant_batch_norm(model, n, m, args)

        if isinstance(m, nn.ReLU) and args.quant_act:
            quant_relu(model, n, m, args, clip_val=a_range[0], device=args.device)
            a_range.pop(0)

    return model



def quantize_model_non_uniform(model, args, w_offsets, a_offsets, w_range, a_range):
    a_offsets = copy.deepcopy(a_offsets.tolist())
    a_range = copy.deepcopy(a_range.tolist())
    w_offsets = copy.deepcopy(w_offsets.tolist())
    w_range = copy.deepcopy(w_range.tolist())

    a_offsets.pop(0)  # first one for the input
    a_range.pop(0)
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            quant_conv(model, n, m, args, w_offsets[0], w_range[0])
            w_offsets.pop(0)
            w_range.pop(0)
        if isinstance(m, nn.Linear):
            quant_linear(model, n, m, args, w_offsets[0], w_range[0])
            w_offsets.pop(0)
            w_range.pop(0)
        if isinstance(m, nn.ReLU):
            quant_relu(model, n, m, args, a_offsets[0], a_range[0], args.device)
            a_range.pop(0)
            a_offsets.pop(0)

    return model


def quantize_model(model, args, w_offsets=(), a_offsets=(), w_range=(), a_range=(), uniform=True):
    if uniform:
        return quantize_model_uniform(model, args, a_range)
    else:
        return quantize_model_non_uniform(model, args, w_offsets, a_offsets, w_range, a_range)


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


class QSGD(torch.optim.SGD):
    def __init__(self, *kargs, **kwargs):
        super(QSGD, self).__init__(*kargs, **kwargs)

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if hasattr(p, 'org'):
                    p.data.copy_(p.org)
        super(QSGD, self).step()
        for group in self.param_groups:
            for p in group['params']:
                if hasattr(p, 'org'):
                    p.org.copy_(p.data)


class QAdam(torch.optim.Adam):
    def __init__(self, *kargs, **kwargs):
        super(QAdam, self).__init__(*kargs, **kwargs)

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if hasattr(p, 'org'):
                    p.data.copy_(p.org)
        super(QAdam, self).step()
        for group in self.param_groups:
            for p in group['params']:
                if hasattr(p, 'org'):
                    p.org.copy_(p.data)


def quantize_uniform(data, n_bits, clip, device='cuda'):
    w_c = data.clamp(-clip, clip)
    b = torch.pow(torch.tensor(2.0), 1 - n_bits).to(device)
    w_q = clip * torch.min(b * torch.round(w_c / (b * clip)), 1 - b)

    return w_q


def quantize_act(data, n_bits, clip, device='cuda'):
    d_c = data.clamp(0, clip)
    b = torch.pow(torch.tensor(2.0), -n_bits).to(device)
    d_q = clip * torch.min(b * torch.round(d_c / (b * clip)), 1 - b)

    return d_q


class QBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, quant_args=None, init_args=None, *kargs, **kwargs):
        super(QBatchNorm2d, self).__init__(*kargs, **kwargs)
        self.weight.data = init_args['weight_data']
        self.bias.data = init_args['bias_data']
        self.clip_val = 0
        self.num_bits_act = quant_args['num_bits_act']

    def forward(self, inputs):
        out = super(QBatchNorm2d, self).forward(inputs)
        relu_clip_val = torch.max(self.bias.data + 3 * self.weight.data.abs()).item()
        out.data = quantize_act(out.data, n_bits=self.num_bits_act, clip=relu_clip_val, device=self.weight.data.device)
        return out


class QConv2d(nn.Conv2d):
    def __init__(self, quant_scheme='TWN', quant_args=None, init_args=None, *kargs, **kwargs):
        super(QConv2d, self).__init__(*kargs, **kwargs)
        self.weight.data = init_args['weight_data']
        if kwargs['bias']:
            self.bias.data = init_args['bias_data']
        self.quant_scheme = quant_scheme
        self.clip_val = 0
        self.num_bits_weight = quant_args['num_bits_weight']
        self.num_bits_act = quant_args['num_bits_act']
        self.quant_act = quant_args['quant_act']
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        if self.bias is not None:
            if not hasattr(self.bias, 'org'):
                self.bias.org = self.bias.data.clone()
        self.quantize_params()

    def forward(self, inputs):

        out = F.conv2d(inputs, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out

    def quantize_params(self):
        clip_val_weight = self.weight.data.view(-1).abs().max()
        clip_val_bias = self.bias.data.view(-1).abs().max() if self.bias is not None else 0
        clip_val = max(clip_val_weight, clip_val_bias)
        clip_val = 2 ** (round(math.log(clip_val, 2)))
        self.weight.data = quantize_uniform(self.weight.data, clip=clip_val, n_bits=self.num_bits_weight,
                                            device=self.weight.data.device)
        if self.bias is not None:
            self.bias.data = quantize_uniform(self.bias.data, clip=clip_val, n_bits=self.num_bits_weight,
                                              device=self.bias.data.device)
        self.clip_val = clip_val


class QLinear(nn.Linear):
    def __init__(self, quant_scheme, quant_args=None, init_args=None, *kargs, **kwargs):
        super(QLinear, self).__init__(*kargs, **kwargs)
        self.weight.data = init_args['weight_data']
        if kwargs['bias']:
            self.bias.data = init_args['bias_data']
        self.quant_scheme = quant_scheme
        self.clip_val = 0
        self.num_bits_weight = quant_args['num_bits_weight']
        self.num_bits_act = quant_args['num_bits_act']
        self.quant_act = quant_args['quant_act']

        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        if self.bias is not None:
            if not hasattr(self.bias, 'org'):
                self.bias.org = self.bias.data.clone()
        self.quantize_params()

    def forward(self, inputs):

        out = F.linear(inputs, self.weight, self.bias)
        return out

    def quantize_params(self):
        clip_val_weight = self.weight.data.view(-1).abs().max()
        clip_val_bias = self.bias.data.view(-1).abs().max() if self.bias is not None else 0
        clip_val = max(clip_val_weight, clip_val_bias)

        clip_val = 2 ** (round(math.log(clip_val, 2)))
        self.weight.data = quantize_uniform(self.weight.data, clip=clip_val, n_bits=self.num_bits_weight,
                                            device=self.weight.data.device)
        if self.bias is not None:
            self.bias.data = quantize_uniform(self.bias.data, clip=clip_val, n_bits=self.num_bits_weight,
                                              device=self.bias.data.device)
        self.clip_val = clip_val


class QReLU(nn.ReLU):
    def __init__(self, bits, clip_val, device, *args, **kwargs):
        super(QReLU, self).__init__(*args, **kwargs)
        self.clip_val = clip_val
        self.bits = bits
        self.device = device

    def forward(self, input):
        act = super(QReLU, self).forward(input)
        actq = quantize_act(act, n_bits=self.bits, clip=self.clip_val, device=self.device)
        return actq
