# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# ------------------------------------------------------------------------------
import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import logging

import yaml
from torch.utils.data import DataLoader

from semseg.models import *
from semseg.datasets import *
from semseg.augmentations import get_val_augmentation
from tools.infer import console, SemSeg

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False


def parse_args():
    parser = argparse.ArgumentParser(description='Speed Measurement')

    parser.add_argument('--a', help='pidnet-s, pidnet-m or pidnet-l', default='pidnet-s', type=str)
    parser.add_argument('--c', help='number of classes', type=int, default=2)
    parser.add_argument('--r', help='input resolution', type=int, nargs='+', default=(1024, 2048))

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    os.chdir('..')
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/custom.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    device = torch.device(cfg['DEVICE'])
    eval_cfg = cfg['EVAL']
    # Comment batchnorms here and in model_utils before testing speed since the batchnorm could be integrated into conv operation
    # (do not comment all, just the batchnorm following its corresponding conv layer)
    model_path = Path(eval_cfg['MODEL_PATH'])
    model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], 2)
    model.load_state_dict(torch.load(str(model_path), map_location='cuda:0'))
    model = model.to(device)
    model.eval()
    iterations = None

    input = torch.randn(1, 3, eval_cfg["IMAGE_SIZE"][0], eval_cfg["IMAGE_SIZE"][1]).cuda()
    with torch.no_grad():
        for _ in range(10):
            model(input)

        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(input)
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)

        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(iterations):
            model(input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    FPS = 1000 / latency
    print(FPS)





