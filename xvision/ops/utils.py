import torch
import torch.nn as nn


def group_parameters(module: nn.Module, bias_decay=0.0):
    with_decay = []
    without_decay = []
    for name, param in module.named_parameters(recurse=True):
        if name.endswith('.bias'):
            without_decay.append(param)
        else:
            with_decay.append(param)
    assert len(list(module.parameters())) == len(without_decay) + \
        len(with_decay), "parameter number inconsistent when grouping parameters"
    return [
        dict(params=with_decay),
        dict(params=without_decay, weight_decay=bias_decay)
    ]


def fuse_model(model: torch.nn.Module):
    # fuse pattern
    # [conv bn relu]
    # [conv bn]
    # [conv relu]
    def _fuse_model(module: torch.nn.Module):
        fuses = []
        names = []
        types = []
        for name, m in module.named_children():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                if len(names) > 1:
                    fuses.append(fuses)
                names, types = [name], ['conv']
            elif isinstance(m, nn.BatchNorm2d):
                if types and types[-1] == 'conv':
                    names.append(name)
                    types.append('bn')
                else:
                    if len(names) > 1:
                        fuses.append(names)
                    names, types = [], []
            elif isinstance(m, nn.ReLU):
                if types and types[-1] in ['conv', 'bn']:
                    names.append(name)
                    types.append('bn')
                if len(names) > 1:
                    fuses.append(names)
                names, types = [], []
            else:
                if len(names) > 1:
                    fuses.append(names)
                names, types = [], []
                _fuse_model(m)

        if fuses:
            torch.quantization.fuse_modules(module, fuses, inplace=True)

    _fuse_model(model)
