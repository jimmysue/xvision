import time
from pathlib import Path


import torch
from torch.utils.data import DataLoader

import xvision.datasets as datasets
import xvision.models.fa as models
from xvision.utils.logger import get_logger
from xvision.utils.saver import Saver
from xvision.ops.nme import IbugScore

from transform import Transform
from train import evaluate

def main(args):
    workdir = Path(args.workdir)
    logger = get_logger()
    logger.info(f'config:\n{args}')
    state = Saver.load_best_from_folder(workdir, map_location='cpu')
  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'use device: {device}')

    num_points = len(args.data.symmetry)
    model = models.__dict__[args.model.name](num_points)
    model.load_state_dict(state['model'])
    model.to(device)
   
    # datasets
    valtransform = Transform(args.dsize, args.padding, args.data.meanshape, args.data.meanbbox)
    valdata = datasets.__dict__[args.data.name](**args.data.val)
    valdata.transform = valtransform
    valloader = DataLoader(valdata, args.batch_size, False,
                           num_workers=args.num_workers, pin_memory=False)

    score_fn = IbugScore(args.left_eye, args.right_eye)
    evalmeter = evaluate(model, valloader, score_fn, device)
    logger.info(evalmeter)

if __name__ == '__main__':
    from config import cfg
    cfg.parse_args()
    cfg.freeze()
    main(cfg)
