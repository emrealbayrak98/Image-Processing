
"""
Created on Mon Mar 27 22:26:34 2023
R-G-B Channel Histograms
@author: erdem
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_uint
from skimage.io import imshow, imread
from skimage.color import rgb2hsv
from skimage.color import rgb2gray
image = imread('img.jpg') #load image
photo_gray = rgb2gray(image) # Convert the input image to grayscale
thresh = 0.35 # Set the threshold value
binary = photo_gray > thresh # Apply thresholding to the image
# Obtain the individual color channels from the input image
red_channel = image[:, :, 0]
green_channel = image[:,:,1]
blue_channel = image[:,:,2]
fig, ax = plt.subplots(1, 3, figsize=(10, 5)) # Create a figure with three
subplots
# Plot the histograms of each channel in a separate subplot
ax[0].hist(red_channel.ravel(), bins=256, color='red')
ax[1].hist(green_channel.ravel(), bins=256, color='green')
ax[2].hist(blue_channel.ravel(), bins=256, color='blue')
# Set titles and axis labels for each subplot
ax[0].set_title('Red Channel')
ax[1].set_title('Green Channel')
ax[2].set_title('Blue Channel')
ax[0].set_xlabel('Value')
ax[1].set_xlabel('Value')
ax[2].set_xlabel('Value')
ax[0].set_ylabel('Frequency')
plt.show() # Show plot
fig, ax = plt.subplots(1, 2, figsize=(10,6), sharey = True) #plot both of
images
ax[0].imshow(photo_gray, cmap='gray')
ax[0].set_title('Grayscaled image')
ax[1].imshow(binary, cmap='gray')
ax[1].set_title('0.35 Thresholded image')