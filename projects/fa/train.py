from os import getloadavg
from pathlib import Path

import torch
from torch.optim import SGD
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import OneCycleLR

from xvision.utils.logger import get_logger
from xvision.models import fa as models
from xvision.ops.utils import group_parameters

def main(args):
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(workdir/'log.txt')
    logger.info(f'config:\n{args}')

    # dump all configues
    with open(workdir / 'config.yml', 'wt') as f:
        args.dump(stream=f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'use device: {device}')

    model = models.__dict__[args.model.name]()

    model.to(device)
    parameters = group_parameters(model, bias_decay=0)
    optimizer = SGD(parameters, args.lr, args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = OneCycleLR(optimizer, max_lr = args.lr, div_factor=20, total_steps = args.total_steps, pct_start=0.1, final_div_factor=100)

    # datasets

    


if __name__ == '__main__':
    from config import cfg
    cfg.parse_args()
    cfg.freeze()
    main(cfg)
