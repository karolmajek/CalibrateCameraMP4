#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import cv2
import sys

def main():
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.


    cap = cv2.VideoCapture(sys.argv[1])

    while(True):
        # Capture frame-by-frame
        ret = False
        for i in range(1):
            ret, frame = cap.read()
        if ret is False:
            break

        img = frame
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (9,6), corners2,ret)
            chess=cv2.resize(img,(0,0),fx=0.5,fy=0.5)
            cv2.imshow('chess',chess)
            cv2.waitKey(1)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

    np.save('mtx.npy',mtx)
    np.save('dist.npy',dist)

    print (mtx)

if __name__ == "__main__":
    if len(sys.argv) !=2:
        print("One argument - path to video file")
        print(sys.argv[0],'video.mp4')
        exit(1)
    main()
