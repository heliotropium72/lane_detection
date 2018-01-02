# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 15:17:47 2017

@author: asd
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2


def _curvature(y, fit_coeff):
    ''' Calculate the curvature at row y for given fit coefficients '''
    numerator = ((1 + (2*fit_coeff[0]*y*Line.ym_per_pix + fit_coeff[1])**2)**1.5)
    denominator = np.absolute(2*fit_coeff[0])
    return numerator / denominator

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
    
    def __init__(self, binary_warped=None, lane_inds=None):
        # was the line detected in the last iteration?
        self.detected = False
        
        #Detected line pixels in x and y direction
        self.lane_x = None  
        self.lane_y = None
        
        #polynomial coefficients and numpy polynome for the most recent fit
        self.current_fit = [np.array([False])]
        self.current_poly = None
        
        # smoothed fit coefficients and numpy polynome
        self.fit_smooth = [np.array([False])]
        self. poly_smooth = None
        
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        # Flag logging direction 'left', 'straight', 'right'
        self.curve = ''
        
        if binary_warped is not None and lane_inds is not None:
            self.set_line_pixel(binary_warped, lane_inds)
            self.fit_lane()
            self.calculate_curvature()

    def set_line_pixel(self, binary_warped, lane_inds):
        ''' set the line pixel from lane indices from a binary image 
        see lanedetection.py for more details '''
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Extract left and right line pixel positions
        self.lane_x = nonzerox[lane_inds]
        self.lane_y = nonzeroy[lane_inds]
    
    def set_line_pixel_previous(self, previous_lines):
        lane_x = []
        lane_y = []
        lane_x.extend(self.lane_x)
        lane_y.extend(self.lane_y)
        
        for line in previous_lines:
            lane_x.extend(line.lane_x)
            lane_y.extend(line.lane_y)
        lane_x = np.array(lane_x)
        lane_y = np.array(lane_y)
        return lane_x, lane_y
    
    def fit_lane(self, binary_warped=None, lane_inds=None):
        ''' Fit a second order poynomial to the line pixel '''
        
        if self.lane_x is None or self.lane_y is None:
            self.set_line_pixel(binary_warped, lane_inds)

        try:
            self.current_fit = np.polyfit(self.lane_y, self.lane_x, 2)
            self.current_poly = np.poly1d(self.current_fit)
            self.detected = True
        except:
            self.detected = False

    def fit_lane_incl_previous(self, previous_lines):
        ''' add all detected lane pixels of previous
        detections and do a single fit'''
        
        lane_x, lane_y = self.set_line_pixel_previous(previous_lines)
            
        self.fit_smooth = np.polyfit(lane_y, lane_x, 2)
        self.poly_smooth = np.poly1d(self.fit_smooth)
        self.detected = True

    def calculate_curvature(self):
        if not self.detected:
            #print('Lane was not detected')
            return
        
        fit_meter = np.polyfit(self.lane_y * Line.ym_per_pix,
                                      self.lane_x * Line.xm_per_pix, 2)
        y_eval = Line.Ny_w
        self.radius_of_curvature = _curvature(y_eval, fit_meter)
        
    def calculate_curvature_incl_previous(self, previous_lines):
        lane_x, lane_y = self.set_line_pixel_previous(previous_lines)
        
        fit_meter = np.polyfit(lane_y * Line.ym_per_pix,
                                      lane_x * Line.xm_per_pix, 2)
        y_eval = Line.Ny_w
        return _curvature(y_eval, fit_meter)

    def check_curvature(self, previous,
                        threshold=300, min_value=100, straight_value=2000):
        ''' Compare the found curvature with a previous curvature '''
        if self.radius_of_curvature > straight_value:
            return True
        if self.radius_of_curvature > min_value and (np.abs(self.radius_of_curvature - previous) < threshold):
            return True
        else:
            return False

class Lane():
    ''' Full lane, containg two line objects '''
    
    def __init__(self, left, right, previous=[None]):
        '''
        Parameter:
        left : Line object
            containing information about the left lane border
        right : Line object
            containing information about the right lane border
        previous: list of Lane objects
            previously detected lanes with the last list element is the
            most recent one
        '''
            
        self.left = left
        self.right = right
        
         # Previous lane detections, where the last element is the most recent one    
        self.previous = previous
        
        self.prev_curv_left = None
        self.prev_curv_right = None
        self.prev_center = None
        #if self.previous[-1] is not None:
        #    self.previous_curvatures()
        
        self.__detected = self.left.detected and self.right.detected
        self.__sanity = False
        self.__smoothed = False
        
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        self.calculate_distance()
    
    @property
    def previous(self):
        return self.__previous
    
    @previous.setter
    def previous(self, value):
        try:
            value[0]
        except:
            raise ValueError('Lane.previous needs a list as input')
            
        self.__previous = value
        if self.__previous[-1] is not None:
            self.previous_curvatures()
    
    @property
    def detected(self):
        self.__detected = self.left.detected & self.right.detected
        return self.__detected

    @detected.setter
    def detected(self, value):
        self.__detected = value
    
    @property
    def sanity(self):
        return self.__sanity

    @sanity.setter
    def sanity(self, value):
        self.__sanity = value
    
    @property
    def smoothed(self):
        return self.__smoothed
    
    @smoothed.setter
    def smoothed(self, value):
        self.__smoothed = value
     
    def calculate_distance(self):
        if not self.detected:
            self.line_base_pos = None
            return
        
        left_base_x = self.left.current_poly(Line.Ny_w)
        right_base_x = self.right.current_poly(Line.Ny_w)
        center_x = Line.Nx_w/2
        car_x = ( right_base_x - left_base_x) / 2 + left_base_x
        self.line_base_pos = (center_x-car_x)*Line.xm_per_pix

    def __check_curvatures(self, threshold=300):
        '''
        Check the curvature detection
        On a straight line, curvature would be inifinity
        On a curved line, curvature should be similar to the previous images
        within a threshold
        '''
        left_success = self.left.check_curvature(previous=self.prev_curv_left)
        right_success = self.right.check_curvature(previous=self.prev_curv_right)
        return (left_success and right_success)

    def check_sanity(self):
        if not self.detected:
            self.__sanity = False
            return
        
        sanity_tmp = True
        
        # Comparison of the two lines
        # Fit coffecients -> parallel lines
        if self.smoothed:
            fit_coeff_diff = self.left.fit_smooth - self.right.fit_smooth
        else:
            fit_coeff_diff = self.left.current_fit - self.right.current_fit
        if np.abs(fit_coeff_diff[0]) > 2e-4 or np.abs(fit_coeff_diff[1]) > 2e-1:
            sanity_tmp = False
            #print('Failed coeffs')
            #print(fit_coeff_diff)
        # Central distance
        if np.abs(self.line_base_pos) > 1:
            sanity_tmp = False
        # Radius
        if self.left.radius_of_curvature < 1000 and self.right.radius_of_curvature < 1000:           
            if (np.abs(self.left.radius_of_curvature - 
                  self.right.radius_of_curvature) > 250):
                sanity_tmp = False
                
        # Comparison with before
        #( Does not work for smooth fit)        
#        if self.previous[-1] is not None:
#            if not self.__check_curvatures():
#                sanity_tmp = False
        self.__sanity = sanity_tmp

    ### Previous elements
    def previous_curvatures(self):
        ''' average of previous curvatures '''
        
        if self.previous[-1] is None:
            print('Warning: Last lane object is none.')
            self.prev_curv_left = None
            self.prev_curv_right = None
            return None, None
        
        curv_left = []
        curv_right = []
        for lane in self.previous:
            if lane is not None:
                curv_left.append(lane.left.radius_of_curvature)
                curv_right.append(lane.right.radius_of_curvature)
        
        #print(np.mean(curv_left), np.mean(curv_right))
        self.prev_curv_left = np.mean(curv_left)
        self.prev_curv_right = np.mean(curv_right)
        return self.prev_curv_left, self.prev_curv_right
    
    ###
    def previous_lines_left(self):
        lines = []
        for lane in self.previous:
            if lane is not None: lines.append(lane.left)
        return lines
    
    def previous_lines_right(self):
        lines = []
        for lane in self.previous:
            if lane is not None: lines.append(lane.right)
        return lines
        
    def smooth_fit(self):
        ''' execute the smooth fit function in the two line objects '''       
        self.left.fit_lane_incl_previous(self.previous_lines_left())
        self.right.fit_lane_incl_previous(self.previous_lines_right())
        self.smoothed = True
    
    ###########################################################################        
    ### Visualisation

    def output_string(self):
        try:
            if self.smoothed:
                prev_left = self.left.calculate_curvature_incl_previous(self.previous_lines_left())
                prev_right = self.right.calculate_curvature_incl_previous(self.previous_lines_right())
            else:
                prev_left = self.prev_curv_left
                prev_right = self.prev_curv_right
            left_curv = 'Left curvature {:.1f}m ({:.1f}m)'.format(self.left.radius_of_curvature, prev_left)
            right_curv = 'Right curvature {:.1f}m ({:.1f}m)'.format(self.right.radius_of_curvature, prev_right)
            center = 'Distance to lane center: {:.2f}m'.format(self.line_base_pos)
        except:
            if self.previous[-1] is not None:
                print('That should not happen')
            left_curv = 'Left curvature {:.1f}m'.format(self.left.radius_of_curvature)
            right_curv = 'Right curvature {:.1f}m'.format(self.right.radius_of_curvature)
            center = 'Distance to lane center: {:.2f}m'.format(self.line_base_pos)
        return [left_curv, right_curv, center]
    
    
    def generate_points(self, method='single'):
        rows = range(Line.Ny_w)
        if method=='single' or self.previous[-1] is None:
            left_fitx = self.left.current_poly(rows)
            right_fitx = self.right.current_poly(rows)
        elif method=='incl_previous':
            left_fitx = self.left.poly_smooth(rows)
            right_fitx = self.right.poly_smooth(rows)
        else:
            print('Only methods single and incl_previous are defined.')
            
        return left_fitx, right_fitx
    
    def warp_lane(self, Minv, method='single'):
        ''' Draw the warped line on a blank (color) image
        To do: draw right and left lane pixel in different colours for debugging
        '''
        color_warp = np.zeros((Line.Ny_w, Line.Nx_w, 3), dtype=np.uint8)
        #warp_zero = np.zeros_like(warped_binary).astype(np.uint8)
        #color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
        if not self.detected:
            return color_warp
        
        # generate points
        rows = range(Line.Ny_w)
        left_lane, right_lane = self.generate_points(method=method)            
        
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_lane, rows]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_lane, rows])))])
        pts = np.hstack((pts_left, pts_right))
    
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (Line.Nx, Line.Ny))
        
        # Add text
        text = self.output_string()
        cv2.putText(newwarp, text[0], org=(25,40), fontFace=0,
                    fontScale=1, color=(0, 255, 0), thickness=2)
        cv2.putText(newwarp, text[1], org=(25,80), fontFace=0,
                    fontScale=1, color=(0, 255, 0), thickness=2)
        cv2.putText(newwarp, text[2], org=(25,120), fontFace=0,
                    fontScale=1, color=(0, 255, 0), thickness=2)
        
        return newwarp
        # Combine the result with the original image
        #result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
        #ax.imshow(result)
    
    def plot_fit(self, ax, method='single'):
        # Fit results
        rows = range(Line.Ny_w)
        left_fitx, right_fitx = self.generate_points(method=method)
        
        ax.plot(left_fitx, rows, color='yellow')
        ax.plot(right_fitx, rows, color='yellow')
    
    def plot_birdseye(self, image=None, ax=None):
        out_img = np.dstack((image, image, image))*255
        
        if not self.detected:
            return out_img
        
        # Fit results
        #rows = range(Line.Ny_w)
        #left_fitx = self.left.current_poly(rows)
        #right_fitx = self.right.current_poly(rows)
        
        # Originally detected points
        out_img[self.left.lane_y, self.left.lane_x] = [255, 0, 0]
        out_img[self.right.lane_y, self.right.lane_x] = [0, 0, 255]
        
        plt.figure()
        ax = plt.axes()
        plt.title(self.output_string())
        plt.imshow(out_img)
        self.plot_fit(ax)
        #plt.plot(left_fitx, rows, color='yellow')
        #plt.plot(right_fitx, rows, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
