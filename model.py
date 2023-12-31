# This file is adapted from https://github.com/facebookresearch/CutLER/blob/077938c626341723050a1971107af552a6ca6697/cutler/demo/demo.py
# The original license file is the file named LICENSE.CutLER in this repo.

import argparse
import multiprocessing as mp
import pathlib
import shlex
import subprocess
import sys

import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image

sys.path.append('CutLER/cutler/')
sys.path.append('CutLER/cutler/demo')

from config import add_cutler_config
from predictor import VisualizationDemo

mp.set_start_method('spawn', force=True)


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_cutler_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Disable the use of SyncBN normalization when running on a CPU
    # SyncBN is not supported on CPU and can cause errors, so we switch to BN instead
    if cfg.MODEL.DEVICE == 'cpu' and cfg.MODEL.RESNETS.NORM == 'SyncBN':
        cfg.MODEL.RESNETS.NORM = 'BN'
        cfg.MODEL.FPN.NORM = 'BN'
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(
        description='Detectron2 demo for builtin configs')
    parser.add_argument(
        '--config-file',
        default=
        'model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN.yaml',
        metavar='FILE',
        help='path to config file',
    )
    parser.add_argument('--webcam',
                        action='store_true',
                        help='Take inputs from webcam.')
    parser.add_argument('--video-input', help='Path to video file.')
    parser.add_argument(
        '--input',
        nargs='+',
        help='A list of space separated input images; '
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        '--output',
        help='A file or directory to save output visualizations. '
        'If not given, will show output in an OpenCV window.',
    )

    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.35,
        help='Minimum score for instance predictions to be shown',
    )
    parser.add_argument(
        '--opts',
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


CONFIG_PATH = 'CutLER/cutler/model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN.yaml'
WEIGHT_URL = 'http://dl.fbaipublicfiles.com/cutler/checkpoints/cutler_cascade_final.pth'


def load_model(score_threshold: float) -> VisualizationDemo:
    model_dir = pathlib.Path('checkpoints')
    model_dir.mkdir(exist_ok=True)
    weight_path = model_dir / WEIGHT_URL.split('/')[-1]
    if not weight_path.exists():
        subprocess.run(shlex.split(f'wget {WEIGHT_URL} -O {weight_path}'))

    arg_list = [
        '--config-file',
        CONFIG_PATH,
        '--confidence-threshold',
        str(score_threshold),
        '--opts',
        'MODEL.WEIGHTS',
        weight_path.as_posix(),
        'MODEL.DEVICE',
        'cuda:0' if torch.cuda.is_available() else 'cpu',
        'DATASETS.TEST',
        '()',
    ]
    args = get_parser().parse_args(arg_list)
    cfg = setup_cfg(args)
    return VisualizationDemo(cfg)


def run_model(image_path: str, score_threshold: float = 0.5) -> np.ndarray:
    model = load_model(score_threshold)
    image = read_image(image_path, format='BGR')
    _, res = model.run_on_image(image)
    return res.get_image()
