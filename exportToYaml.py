#!/usr/bin/env python
# -*- coding: utf-8 -*-

from subprocess import call
from pathlib import Path
from tqdm import tqdm
from glob import glob
import numpy as np
import shutil
import errno
import time
import sys
import cv2
import os

mtx = np.load('mtx.npy')
dist = np.load('dist.npy')

if __name__ == "__main__":
    if len(sys.argv) !=2:
        print("Usage:\n\t%s sample.jpg"%sys.argv[0])
        exit(-1)
    img = cv2.imread(sys.argv[1])
    h,w,_ = img.shape
    print('image_width:', w)
    print('image_height:', h)
    print('camera_name: pointgrey')
    print('camera_matrix:')
    print('  rows: 3')
    print('  cols: 3')
    print('  data: [', ', '.join(["%.6f"%_ for _ in mtx.flatten()]),']')
    print('distortion_model: plumb_bob')
    print('distortion_coefficients:')
    print('  rows: 1')
    print('  cols: 5')
    print('  data: [', ', '.join(["%.6f"%_ for _ in dist.flatten()]),']')
    print('rectification_matrix:')
    print('  rows: 3')
    print('  cols: 3')
    print('  data: [1, 0, 0, 0, 1, 0, 0, 0, 1]')
    print('projection_matrix:')
    print('  rows: 3')
    print('  cols: 4')
    print('  data: [', ', '.join(["%.6f"%_ for _ in mtx.flatten()]),']')
