# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 14:26:12 2017

@author: asd
"""

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

import lane as l

#importing own functions
try:
    DIST = dist.mtx
    MTX = dist.mtx
except:
    import undistort as dist
    DIST = dist.mtx
    MTX = dist.mtx


import os
'''
os.listdir("test_images/")
test_nr = 4

test_image_path_ls = os.listdir("test_images/")
test_image_path = os.path.join("test_images", test_image_path_ls[test_nr])

test_image_init = mpimg.imread(test_image_path)
'''
import glob
folder_test = 'test_images'
image_files = glob.glob(os.path.join(folder_test,'*.jpg'))
images = []

for fname in image_files:
    images.append(mpimg.imread(os.path.join(fname)))
    
fig, axs = plt.subplots(4,2, figsize=(8, 8))
fig.subplots_adjust(hspace = .1, wspace=.1)
axs = axs.ravel()
for i in range(8):
    axs[i].axis('off')
    axs[i].imshow(images[i])
    

###############################################################################
# Parameter for pipeline

#Birdseye view
s = images[0].shape
src = np.float32([[600,445], [679,445], [247,680], [1057,680]])
x_o = 150 # offset
dst = np.float32([[x_o,0], [s[1]-x_o,0], [x_o,s[0]], [s[1]-x_o,s[0]]])

M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

# Conversion to meter on bottom line
# in the US, lane lines are 3.7 meters wide (=bottom line)
# and the dashed lines are 3 meters long.
# The length of lane is approx. 30 meters
# Define conversions in x and y from pixels space to meters for the bottom line
# of the warped images
lane_length_pix = dst[2][1] - dst[0][1]
lane_width_pix = dst[3][0] - dst[2][0]
ym_per_pix = 30 / lane_length_pix # meters per pixel in y dimension (about 30m long)
xm_per_pix = 3.7 / lane_width_pix # meters per pixel in x dimension (lanes are 3.7m wide in US)

l.Line.Ny = s[0]
l.Line.Nx = s[1]
l.Line.Ny_w = s[0]
l.Line.Nx_w = s[1]
l.Line.ym_per_pix = ym_per_pix
l.Line.xm_per_pix = xm_per_pix

###############################################################################
# Functions for pipeline
    
def RGB_to_HLS(image):
    ''' H (hue), L (lightness), S (saturation) '''
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float)

def hue(RGBimage):
    hls = RGB_to_HLS(RGBimage)
    return hls[:,:,0]

def lightness(RGBimage):
    hls = RGB_to_HLS(RGBimage)
    return hls[:,:,1]    

def saturation(RGBimage):
    hls = RGB_to_HLS(RGBimage)
    return hls[:,:,2]   

def binary_threshold(image, thresh=(0,255)):
    binary = np.zeros_like(image)
    binary[(image >= thresh[0]) & (image <= thresh[1])] = 1
    return binary

def sobel(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    return scaled_sobel

def birdseye(image, src, dst):
    s = image.shape
    #o = 200 # offset
    #dst = np.float32([[o,o], [s[1]-o,o], [o,s[0]], [s[1]-o,s[0]]])
    M = cv2.getPerspectiveTransform(src, dst)
    #Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(image, M, (s[1],s[0]), flags=cv2.INTER_LINEAR)
    return warped

def draw_roi(image, pts):
    cv2.line(image, tuple(pts[0]), tuple(pts[1]), color=[255, 0, 0], thickness=2)
    cv2.line(image, tuple(pts[0]), tuple(pts[2]), color=[255, 0, 0], thickness=2)
    cv2.line(image, tuple(pts[1]), tuple(pts[3]), color=[255, 0, 0], thickness=2)
    cv2.line(image, tuple(pts[2]), tuple(pts[3]), color=[255, 0, 0], thickness=2)
    return image


    
###############################################################################
# Sliding window approach for lane detection in warped image

def _sliding_window(image, n_windows, x_margin, minpix, show=False):
    
    # Derived parameters
    Ny, Nx = image.shape
    window_height = np.int(Ny/n_windows)
    if show: # Create an output image to draw on and  visualize the result
        out_img = np.dstack((image, image, image))*255

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Find the start point at the bottom line using a histogram approach
    # Take a histogram of the bottom half of the image
    histogram = np.sum(image[Ny/2:,:], axis=0)
    # The histogram peaks in the left and right half are the start points of the lanes
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(n_windows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = Ny - (window+1)*window_height
        win_y_high = Ny - window*window_height
        win_xleft_low = leftx_current - x_margin
        win_xleft_high = leftx_current + x_margin
        win_xright_low = rightx_current - x_margin
        win_xright_high = rightx_current + x_margin
            
        if show:
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low,win_y_low),
                              (win_xleft_high,win_y_high), (0,255,0), 2) 
            cv2.rectangle(out_img, (win_xright_low,win_y_low),
                              (win_xright_high,win_y_high), (0,255,0), 2)
    
        # Identify the nonzero pixels in x and y within the window
        def _inside_window(y, x, y_low, y_high, x_low, x_high):
            ''' returns boolean mask for (y,x) inside window '''
            return ((y >= y_low) & (y < y_high) & (x >= x_low) &  (x < x_high))
        def inside_window(lane):
            if lane == 'left':
                return _inside_window(nonzeroy, nonzerox, win_y_low, win_y_high,
                                           win_xleft_low, win_xleft_high)
            elif lane == 'right':
                return _inside_window(nonzeroy, nonzerox, win_y_low, win_y_high,
                                           win_xright_low, win_xright_high)
            else:
                print('Invalid input. Select either left or right for lane argument.')
            
        good_left_inds = inside_window('left').nonzero()[0]
        good_right_inds = inside_window('right').nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    if show:
        return left_lane_inds, right_lane_inds, out_img
    else:
        return left_lane_inds, right_lane_inds, None
    

def sliding_window(image, n_windows=9, x_margin=100, minpix=50, first=True,
                   previous_lane=None, show=True):
    '''
    Parameters
    ----------
    image : np.array
        binary and warped image containing the lane lines
    n_windows : int
        Number of sliding windows in which image is separated along y-axis
    x_margin : int
        width of the window to be addes at either side of the central point
    minpix : int
        minimum bumber of pixels found to recenter window
    '''
    # Derived parameters
    Ny, Nx = image.shape
    if show: # Create an output image to draw on and  visualize the result
        out_img = np.dstack((image, image, image))*255

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    if not first:
        # Initial position of the lane lines are remembered in two fit polynoms
        try:
            left_x_pred = previous_lane.left.current_poly(nonzeroy)
            right_x_pred = previous_lane.right.current_poly(nonzeroy)
            
            # Note to myself: & (bitwise and) and and (logical and) are not the same!
            # Only search +- margin around predicted lane center from previous line fit
            left_lane_inds = np.bitwise_and(nonzerox > (left_x_pred - x_margin),
                                             nonzerox < (left_x_pred + x_margin))
            right_lane_inds = np.bitwise_and(nonzerox > (right_x_pred - x_margin),
                                             nonzerox < (right_x_pred + x_margin))
        except:
            #raise ValueError
            print('Polynomes are not defined. Set polynoms or use first=True')
        
    else:
        left_lane_inds, right_lane_inds, out_img = _sliding_window(image, n_windows, x_margin, minpix, show=show)


    left = l.Line()
    left.fit_lane(image, left_lane_inds)
    left.calculate_curvature()

    right = l.Line()
    right.fit_lane(image, right_lane_inds)
    right.calculate_curvature()

    lane = l.Lane(left, right)
    #lane.plot(image)
    """
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Functional form of the polynomials
    left_fit_poly = np.poly1d(left_fit) 
    right_fit_poly = np.poly1d(right_fit)


    if show:
        plt.figure()
        # Fit results
        rows = range(Ny)
        left_fitx = left_fit_poly(rows)
        right_fitx = right_fit_poly(rows)
        # Originally detected points
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        if first:
            plt.imshow(out_img)
        else:
            window_img = np.zeros_like(out_img)          
            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            left_line_window1 = np.array([np.transpose(
                                    np.vstack([left_fitx-x_margin, rows]))])
            left_line_window2 = np.array([np.flipud(np.transpose(
                                    np.vstack([left_fitx+x_margin, rows])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(
                                    np.vstack([right_fitx-x_margin, rows]))])
            right_line_window2 = np.array([np.flipud(np.transpose(
                                    np.vstack([right_fitx+x_margin, rows])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))
            
            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
            result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
            plt.imshow(result)
        
        plt.plot(left_fitx, rows, color='yellow')
        plt.plot(right_fitx, rows, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
    """    
    return lane #left_fit_poly, right_fit_poly

###############################################################################
# Combine everything to a pipeline
import copy
    
def pipeline(image, s_thresh=(170, 255), sx_thresh=(20, 100),
             previous_lane=None, reset=False, show=False):
    img_o = np.copy(image)
    
    img = dist.undistort(img_o)
    
    img_s = saturation(img)
    binary_s = binary_threshold(img_s, s_thresh)
    
    #img_h = hue(img)
    img_sobel = sobel(img_s)
    binary_sobel = binary_threshold(img_sobel, sx_thresh)
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(binary_sobel)
    combined_binary[(binary_s == 1)]=1# | (binary_sobel == 1)] = 1

    warped_binary = birdseye(combined_binary, src, dst)
    
    if reset:
        lane = sliding_window(warped_binary, first=True, show=False)
    else:
        lane = sliding_window(warped_binary, previous_lane=previous_lane,
                              first=False, show=False)

    ### Do some checks
    ### Save values

    if show:
        fig, axs = plt.subplots(3,3, figsize=(8, 8))
        #fig.subplots_adjust(hspace = .1, wspace=.1)
        #plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        axs = axs.ravel()
        axs[0].set_title('Original image')
        axs[0].imshow(image)
        axs[1].set_title('Undistorted image')
        img_e = copy.copy(img)
        img_e = draw_roi(img_e, src)
        axs[1].imshow(img_e)
        axs[2].set_title('Saturation channel')
        axs[2].imshow(img_s, cmap='Greys_r')
        axs[3].set_title('Saturation channel thresholded')
        axs[3].imshow(binary_s, cmap='Greys_r')
        axs[4].set_title('Sobel of saturation channel')
        axs[4].imshow(img_sobel, cmap='Greys_r')
        axs[5].set_title('Sobel thresholded')
        axs[5].imshow(binary_sobel, cmap='Greys_r')
        axs[6].set_title('Result')
        axs[6].imshow(combined_binary, cmap='Greys_r')
        axs[7].set_title('Result Birdseye')
        axs[7].imshow(warped_binary, cmap='Greys_r')
        lane.plot_fit(axs[7])
        
        axs[8].set_title('Result')
        #warped = birdseye(img_o, src, dst)
        #warped = draw_roi(warped, dst)
        
        undist_with_lane = cv2.addWeighted(img, 1, lane.warp_lane(Minv), 0.3, 0)
        axs[8].imshow(undist_with_lane)

        for i in range(9):
            axs[i].axis('off')
            
        fig.tight_layout()
        
    return lane

###############################################################################
# Apply the functions

result = pipeline(images[4], s_thresh=(120,255), sx_thresh=(20,100),
                  reset=True, show=True)
"""
lane_first = sliding_window(result, first=True, show=False)

lane_new = sliding_window(result, previous_lane=lane_first,
                          first=False, show=False)
"""
#Minv = cv2.getPerspectiveTransform(dst, src)
#plot_result(images[4], result, Minv, left_poly2, right_poly2)

# Test on all test images
fig, axs = plt.subplots(4,2, figsize=(8, 8))
fig.subplots_adjust(hspace = .1, wspace=.1)
axs = axs.ravel()
for i in range(8):
    axs[i].axis('off')
    #axs[i].imshow(images[i])
    lane = pipeline(images[i], s_thresh=(120,255), sx_thresh=(20,100),
                    reset=True, show=False)
    img = dist.undistort(images[i])  
    undist_with_lane = cv2.addWeighted(img, 1, lane.warp_lane(Minv), 0.3, 0)
    axs[i].imshow(undist_with_lane)