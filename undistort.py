# -*- coding: utf-8 -*-
"""
Script for camera calibration

Created on Fri Dec 15 13:37:27 2017

@author: asd
"""


import cv2

import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from os.path import join

folder_cal = 'camera_cal'

# prepare object points
nx = 9
ny = 6

objp = np.zeros((nx*ny,3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
images = glob.glob(join(folder_cal,'*.jpg'))

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
#        img = cv2.drawChessboardCorners(img, (nx,ny), corners2, ret)
#        cv2.imshow('img',img)
#        cv2.waitKey(500)
    else:
        print('Corners were not detected for {}'.format(fname))

cv2.destroyAllWindows()

# Camera calibration
def calibrateCamera():
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape, None, None)
    return mtx, dist

mtx, dist = calibrateCamera()

#print(mtx)
#print(dist)


def undistort(img, mtx=mtx, dist=dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

def undistort_and_warp(img, nx=nx, ny=ny, mtx=mtx, dist=dist):
    ''' Undistort the image and do a perspective transform'''
    # 1) Undistort using mtx and dist
    img = undistort(img, mtx, dist)
    # 2) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 3) Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    # 4) If corners found:
    if ret:
        # a) draw corners
        img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        # b) 4 outermost corners
        src = np.float32([corners[0],corners[nx-1],corners[-nx],corners[-1]])
        # c) define 4 destination points
        rows, cols, depth = img.shape
        dst = np.float32([[0,0],[cols,0],[0,rows],[cols,rows]])
        # d) get M, the transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # e) warp your image to a top-down view
        warped = cv2.warpPerspective(img, M, (cols,rows), flags=cv2.INTER_LINEAR)
        return warped, M
    else:
        print('Corners were not detected')
        return img, None

if __name__ == "__main__":
    img = cv2.imread(images[13])
    img_u = undistort(img)
    img_w, M = undistort_and_warp(img)
    
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image')
    ax2.imshow(img_u)
    ax2.set_title('Undistorted Image')
    ax3.imshow(img_w)
    ax3.set_title('Undistorted and warped Image')
    plt.tight_layout()
    #plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
