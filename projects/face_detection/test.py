import argparse
import torch
import cv2
import tqdm
from pathlib import Path

from xvision.model import fd as MODELS, initialize_model
from xvision.utils import Saver
from xvision.ops import BBoxAnchors


def main(args):
    model_args = args.test.model.args
    model_kwargs = args.test.model.kwargs
    model_name = args.test.model.name
    model = initialize_model(MODELS, model_name, model_args, model_kwargs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.device is not None:
        device = torch.device(torch.device)

    state = Saver.load_best_from_folder(args.workdir, map_location='cpu')
    model.load_state_dict(state['model'])

    model = model.to(device)
    model.eval()

    prior = BBoxAnchors(args.dsize, args.strides, args.fsizes, args.layouts,
                        args.iou_threshold, args.encode_mean, args.encode_std)

    for path in tqdm.tqdm(args.test.image_dir.rglob(f'*.jpg')):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)

        # resize image with short size as config
        h, w = image.shape[:2]
        scale = max(args.test.short_size / h, args.test.short_size / w)
        dh, dw = int(scale * h), int(scale * w)

        

if __name__ == '__main__':
    from config import cfg
    cfg.parse_args()
    cfg.freeze()
    main(cfg)
