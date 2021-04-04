import torch
from pathlib import Path


def main(args):
    workdir = args.workdir
    pass

if __name__ == '__main__':
    from config import cfg
    cfg.parse_args()
    cfg.freeze()
    main(cfg)
