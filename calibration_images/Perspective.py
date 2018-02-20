#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 15:56:56 2018

@author: oktanto
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
# Uncomment the next line for use in a Jupyter notebook
# This enables the interactive matplotlib window
#%matplotlib notebook
image = mpimg.imread('example_rock4.jpg')

def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped

def color_thresh(img, rgb_thresh_bellow=(160, 160, 160), rgb_thresh_up=(255, 255, 255)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh_bellow[0]) \
                & (img[:,:,0] < rgb_thresh_up[0] + 1) \
                & (img[:,:,1] > rgb_thresh_bellow[1]) \
                & (img[:,:,1] < rgb_thresh_up[1] + 1) \
                & (img[:,:,2] > rgb_thresh_bellow[2]) \
                & (img[:,:,2] < rgb_thresh_up[1] + 1)
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

# Define calibration box in source (actual) and destination (desired) coordinates
# These source and destination points are defined to warp the image
# to a grid where each 10x10 pixel square represents 1 square meter
# The destination box will be 2*dst_size on each side
dst_size = 5 
# Set a bottom offset to account for the fact that the bottom of the image 
# is not the position of the rover but a bit in front of it
# this is just a rough guess, feel free to change it!
bottom_offset = 2
source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset], 
                  [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                  ])
    
images = perspect_transform(image, source, destination)

hsv_rock = cv2.cvtColor(images, cv2.COLOR_BGR2HSV)

# define range of yellow color in HSV
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([140, 255, 255])

threshed_rock = cv2.inRange(hsv_rock, lower_yellow, upper_yellow)

plt.subplot(221)
plt.imshow(image)
plt.subplot(222)
plt.imshow(hsv_rock)
plt.subplot(223)
plt.imshow(threshed_rock/255)
plt.show()