# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 15:17:47 2017

@author: asd
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Define a class to receive the characteristics of each line detection
class Line():
    
    # These parameters are the same for all class objects
    xm_per_pix = None
    ym_per_pix = None
    
    # Original image
    Ny = None
    Nx = None
    # Warped image
    Ny_w = None
    Nx_w = None
    
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.lane_x = None  
        #y values for detected line pixels
        self.lane_y = None
        
    def fit_lane(self, binary_warped, lane_inds):
        
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Extract left and right line pixel positions
        self.lane_x = nonzerox[lane_inds]
        self.lane_y = nonzeroy[lane_inds] 

        # Fit a second order polynomial to each
        self.current_fit = np.polyfit(self.lane_y, self.lane_x, 2)
        self.current_poly = np.poly1d(self.current_fit)

    def _curvature(self, y, fit_coeff):
            numerator = ((1 + (2*fit_coeff[0]*y*Line.ym_per_pix + fit_coeff[1])**2)**1.5)
            denominator = np.absolute(2*fit_coeff[0])
            return numerator / denominator

    def calculate_curvature(self):
        fit_meter = np.polyfit(self.lane_y * Line.ym_per_pix,
                                      self.lane_x * Line.xm_per_pix, 2)
        y_eval = Line.Ny_w
        self.radius_of_curvature = self._curvature(y_eval, fit_meter)

class Lane():
    ''' Full lane, containg two line objects '''
    
    def __init__(self, left, right):
        self.detected = False
        
        self.left = left
        self.right = right
        
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        self.calculate_distance()
        
    def calculate_distance(self):
        left_base_x = self.left.current_poly(Line.Ny_w)
        right_base_x = self.right.current_poly(Line.Ny_w)
        center_x = Line.Nx_w/2
        car_x = ( right_base_x - left_base_x) / 2 + left_base_x
        self.line_base_pos = (center_x-car_x)*Line.xm_per_pix

    def check_sanity(self):
        sanity = True
        if np.abs(self.line_base_pos) > 1:
            sanity = False
        if (np.abs(self.left.radius_of_curvature - 
                  self.right.radius_of_curvature) > 500):
            sanity = False
        self.detected = sanity

    ### Visualisation

    def output_string(self):
        left_curv = 'Left curvature {:.1f}m'.format(self.left.radius_of_curvature)
        right_curv = 'Right curvature {:.1f}m'.format(self.right.radius_of_curvature)
        center = 'Distance to lane center: {:.2f}m'.format(self.line_base_pos)
        return left_curv + '\n' + right_curv + '\n' + center
    
    
    def warp_lane(self, Minv):
        color_warp = np.zeros((Line.Ny_w, Line.Nx_w, 3), dtype=np.uint8)
        #warp_zero = np.zeros_like(warped_binary).astype(np.uint8)
        #color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
        # generate points
        rows = range(Line.Ny_w)
        left_lane = self.left.current_poly(rows)
        right_lane = self.right.current_poly(rows)
        
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_lane, rows]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_lane, rows])))])
        pts = np.hstack((pts_left, pts_right))
    
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (Line.Nx, Line.Ny)) 
        return newwarp
        # Combine the result with the original image
        #result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
        #ax.imshow(result)
    
    def plot_fit(self, ax):
        # Fit results
        rows = range(Line.Ny_w)
        left_fitx = self.left.current_poly(rows)
        right_fitx = self.right.current_poly(rows)
        
        ax.plot(left_fitx, rows, color='yellow')
        ax.plot(right_fitx, rows, color='yellow')
    
    def plot(self, image=None, ax=None):
        out_img = np.dstack((image, image, image))*255
        
        # Fit results
        rows = range(Line.Ny_w)
        left_fitx = self.left.current_poly(rows)
        right_fitx = self.right.current_poly(rows)
        
        # Originally detected points
        out_img[self.left.lane_y, self.left.lane_x] = [255, 0, 0]
        out_img[self.right.lane_y, self.right.lane_x] = [0, 0, 255]
        
        plt.figure()
        plt.title(self.output_string())
        plt.imshow(out_img)
        plt.plot(left_fitx, rows, color='yellow')
        plt.plot(right_fitx, rows, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
