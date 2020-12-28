import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn


from nn_inference import precision_profiler, quantization


class QuantInput(object):
    def __init__(self, clip_val, num_bits, device='cpu'):
        self.num_bits = num_bits
        self.clip_val = clip_val
        self.device = device

    def __call__(self, data):
        data = quantization.quantize_uniform(data, n_bits=self.num_bits, clip=self.clip_val, device='cpu')
        return data


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = ProgressMeter._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print2(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    @classmethod
    def _get_batch_fmtstr(cls, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self.nclass = num_classes

    def forward(self, x):
        conv_features = self.features(x)
        flatten = conv_features.view(conv_features.size(0), -1)
        fc = self.fc_layers(flatten)
        return fc


class QuantArgs:
    def __init__(self, *args, **kwargs):
        self.num_bits_act = kwargs.get('num_bits_act')
        self.quant_act = kwargs.get('quant_act')
        self.num_bits_weight = kwargs.get('num_bits_weight')
        self.quant_scheme_conv = kwargs.get('quant_scheme_conv')
        self.quant_scheme_fc = kwargs.get('quant_scheme_fc')
        self.device = kwargs.get('device')


def get_datasets(*args, **kwargs):
    trainset = torchvision.datasets.CIFAR10(train=True, *args, **kwargs)
    testset = torchvision.datasets.CIFAR10(train=False, *args, **kwargs)
    return trainset, testset


def get_dataloaders(trainset, testset, batch_size=100, num_worker=4):
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_worker)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_worker)

    return trainloader, testloader


def get_model(model_src_path, device='cpu'):
    model = Net(num_classes=10)
    state_dict = torch.load(model_src_path, map_location=device)
    model.load_state_dict(state_dict)
    return model


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        if type(output) is tuple:
            _, _, output = output
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res, pred[0, :]


def eval_single_batch_compute(x, y, model):
    output = model(x)
    accs, predictions = accuracy(output, y, topk=(1,))
    acc = accs[0]
    return acc, predictions


def eval_model(model, dataloader, print_acc=False, device='cpu', log_update_feq=20):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(dataloader),
        [top1],
        prefix='Evaluating Batch'
    )

    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            x, y = data
            x.requires_grad = True
            x = x.to(device)
            y = y.to(device)
            n_data = y.size(0)

            acc, predictions = eval_single_batch_compute(x, y, model)

            top1.update(acc.item(), n_data)
            if idx % log_update_feq == log_update_feq - 1:
                progress.print2(idx + 1)

        if print_acc:
            print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


def feedforward():
    device = 'cpu'
    print('using device:', device)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    trainset, testset = get_datasets(root='./data', download=True,transform=transform)
    _, testloader = get_dataloaders(trainset, testset, batch_size=100, num_worker=16)

    model_src_path = 'model.tar'  # todo you need to set the path to downloaded model !!
    model = get_model(model_src_path, device)
    model = model.to(device)
    eval_model(model, testloader, print_acc=True, device=device)


def compute_precision_offsets():
    device = 'cuda'
    print('using device:', device)
    trainset, testset = get_datasets(root='./data', download=True)
    trainloader, testloader = get_dataloaders(trainset, testset, batch_size=500, num_worker=32)

    model_src_path = 'model.tar'
    model = get_model(model_src_path, device)
    model = model.to(device)
    precision_profiler.GradCollect.retain_model_grad(model)
    precision_profiler.GradCollect.retain_inputs_grad(model)

    wg, ag = precision_profiler.get_noise_gains(model, trainloader, device)
    wg, ag, least_gain = precision_profiler.get_normalized_noise_gains(wg, ag)
    #print(precision_profiler.GradCollect.weight_range_collect)
    #print(precision_profiler.GradCollect.activation_range_collect)
    w_offsets, a_offsets = precision_profiler.get_precision_offsets(wg, ag, least_gain)
    #print(w_offsets, a_offsets)
    
    #save the results
    np.save(arr=w_offsets, file='weight_offsets.npy')
    np.save(arr=a_offsets, file='activation_offsets.npy')
    np.save(arr=precision_profiler.GradCollect.activation_range_collect, file='activation_dynamic_range.npy')
    np.save(arr=np.array([v for k, v in precision_profiler.GradCollect.weight_range_collect]),
            file='weight_dynamic_range.npy')


def qfeedforward():
    device = 'cuda'
    print('using device:', device)

    w_offsets = np.load('weight_offsets.npy')
    a_offsets = np.load('activation_offsets.npy')
    w_range = np.load('weight_dynamic_range.npy')
    a_range = np.load('activation_dynamic_range.npy')

    num_bits = 5 # Bmin
    print(num_bits+w_offsets)
    print(num_bits+a_offsets)
    args = QuantArgs(
        num_bits_act=num_bits,
        num_bits_weight=num_bits,
        quant_act=True,
        quant_scheme_fc='float',
        quant_scheme_conv='float',
        device=device
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            QuantInput(clip_val=a_range[0],num_bits=num_bits, device=device)
        ]
    )

    trainset, testset = get_datasets(root='./data', download=True, transform=transform)
    _, testloader = get_dataloaders(trainset, testset, batch_size=100, num_worker=16)

    model_src_path = 'model.tar'
    model = get_model(model_src_path, device)
    model = model.to(device)

    model = quantization.quantize_model(
        model, args,
        w_offsets=w_offsets,
        w_range=w_range,
        a_offsets=a_offsets,
        a_range=a_range,
        uniform=False
    )

    eval_model(model, testloader, print_acc=True, device=device)


if __name__ == '__main__':
    #feedforward()
    qfeedforward() # quantized inference
    #compute_precision_offsets() # computing precision offsets
