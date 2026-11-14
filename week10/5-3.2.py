import cv2 as cv
import numpy as np
import time
import os

# Read images
img1 = cv.imread('mot_color70.jpg')
img2 = cv.imread('mot_color83.jpg')

# Check image loading
if img1 is None:
    print("Error: mot_color70.jpg not found!")
    exit()

if img2 is None:
    print("Error: mot_color83.jpg not found!")
    exit()

# Crop bus from the first image
img1 = img1[190:350, 440:560]

# Convert to grayscale
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# Create SIFT detector
sift = cv.SIFT_create()

# Detect keypoints and descriptors
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# Print number of keypoints
print('Number of keypoints:', len(kp1), len(kp2))
# Start matching time
start = time.time()

# Create FLANN matcher
flann_matcher = cv.DescriptorMatcher_create(
    cv.DescriptorMatcher_FLANNBASED
)

# Find 2 nearest matches
knn_match = flann_matcher.knnMatch(des1, des2, 2)

# Apply nearest neighbor distance ratio
T = 0.7
good_match = []

for nearest1, nearest2 in knn_match:
    if nearest1.distance / nearest2.distance < T:
        good_match.append(nearest1)

# Print matching time
print('Matching time:', time.time() - start)

# Create empty image for matching results
img_match = np.empty(
    (max(img1.shape[0], img2.shape[0]),
     img1.shape[1] + img2.shape[1], 3),
    dtype=np.uint8
)

# Draw good matches
cv.drawMatches(
    img1, kp1,
    img2, kp2,
    good_match,
    img_match,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# Display results
cv.imshow('Good Matches', img_match)

k = cv.waitKey(0)
cv.destroyAllWindows()