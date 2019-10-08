import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# prepare object points
nx = 8#TODO: enter the number of inside corners in x
ny = 6#TODO: enter the number of inside corners in y

# Make a list of calibration images
fname = 'calibration_test.png'
img = cv2.imread(fname)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

# If found, draw corners
#if ret == True:
    # Draw and display the corners
   # cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
    #plt.imshow(img)

plt.imshow(img)
plt.plot(corners[0][0][0],corners[0][0][1],'.')
plt.plot(corners[7][0][0],corners[7][0][1],'.')
plt.plot(corners[40][0][0],corners[40][0][1],'.')
plt.plot(corners[47][0][0],corners[47][0][1],'.')
plt.imshow(img)
print(corners[0][0][0])