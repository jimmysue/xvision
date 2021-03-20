from pathlib import Path
from xvision.utils.logger import get_logger
from xvision.utils.saver import Saver
from xvision.utils.meter import MetricLogger


class Trainer(object):
    def __init__(self, args) -> None:
        super().__init__()



def prepare_workspace(args):
    workdir = args.workdir