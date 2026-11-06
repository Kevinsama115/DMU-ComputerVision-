import cv2 as cv
import numpy as np
import time

# Read images
img1 = cv.imread('mot_color70.jpg')[190:350, 440:560]  # Bus cropped as model image
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

img2 = cv.imread('mot_color83.jpg')  # Scene image
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# Create SIFT detector
sift = cv.SIFT_create()

# Detect keypoints and descriptors
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# Print number of keypoints
print('Number of keypoints:', len(kp1), len(kp2))