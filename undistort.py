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

mapx = None
mapy = None
newcameramtx = None

print('mtx:',mtx)
print('dist:',dist)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def convert_image(img, mtx, dist):
    global mapx,mapy,newcameramtx
    # dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    if mapx is None:
        h,  w = img.shape[:2]
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
        mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (img.shape[1], img.shape[0]), 5)
        print(newcameramtx)
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    return dst

def convert(input_video, output_video):
    cap = cv2.VideoCapture(input_video)

    tmpdir = "/tmp/"+str(time.time())

    mkdir_p(tmpdir)
    counter=0

    # cv2.initUndistortRectifyMap(cameraMatrix, distCoeffs, R, newCameraMatrix, size, m1type


    mapx = None
    mapy = None
    newcameramtx = None

    time_start = time.clock()

    while(True):
        # Capture frame-by-frame
        ret = False
        ret, frame = cap.read()
        if ret is False:
            break
        if mapx is None:
            mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(frame.shape[1],frame.shape[0]),5)
            time_start = time.clock()

        # cv2.imshow('frame',frame)
        # cv2.waitKey(1)
        img = frame
        # imgsmall=cv2.resize(img,(0,0),fx=0.3,fy=0.3)
        # dst = cv2.undistort(img, mtx, dist, None)
        dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

        # cv2.imshow('dst',dst)
        # cv2.waitKey(1)

        # dstsmall=cv2.resize(dst,(0,0),fx=0.3,fy=0.3)
        # cv2.imshow('imgsmall',np.concatenate((imgsmall,dstsmall),axis=1))
        # cv2.waitKey(1)

        fname = tmpdir+"/%08d.jpg"%counter
        counter = counter + 1
        cv2.imwrite(fname,dst)
        if int(counter)%1000==0:
            print(counter / (time.clock()-time_start))

    ffmpeg=['ffmpeg', '-pattern_type', 'glob', '-i', tmpdir+'/*.jpg', '-i', input_video, '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'libx264', '-r', "30",output_video]
    print(' '.join(ffmpeg))
    call(ffmpeg)
    shutil.rmtree(tmpdir)
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) !=3 and len(sys.argv) !=4 and len(sys.argv) !=5:
        print("Usage:\n\t%s  input.mp4  output.mp4"%sys.argv[0])
        print("OR")
        print("Usage:\n\t%s  dir  input_dir  output_dir"%sys.argv[0])
        print("OR")
        print("Usage:\n\t%s  dir  jpg  input_dir  output_dir"%sys.argv[0])
        exit(-1)
    if len(sys.argv) == 3:
        print("Undistorting video %s => %s"%(sys.argv[1],sys.argv[2]))
        convert(sys.argv[1],sys.argv[2])
    if len(sys.argv) ==4:
        input_path = sys.argv[2]+'/*.MP4'
        print(input_path)
        for fname in tqdm(list(glob(input_path))):
            print("Processing file %s"%fname)
            output = sys.argv[3] + "/" + fname.split('/')[-1]
            # print("Undistorting video %s => %s"%(fname,output))
            if not Path(output).exists():
                convert(fname, output)
    if len(sys.argv) == 5:
        images = glob(sys.argv[3]+'/*.jpg')
        for fname in tqdm(images):
            img = cv2.imread(fname)
            converted = convert_image(img, mtx, dist)
            fnameout = sys.argv[4] + '/' + fname.split('/')[-1]
            # print(fnameout)
            cv2.imwrite(fnameout, converted)
    print('mtx')
    print(mtx)
    print('newcameramtx')
    print(newcameramtx)
