import cv2 as cv

# Read image
img = cv.imread('mot_color70.jpg')

# Convert to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Create SIFT detector
sift = cv.SIFT_create()

# Detect keypoints and descriptors
kp, des = sift.detectAndCompute(gray, None)

# Draw keypoints
gray = cv.drawKeypoints(
    gray,
    kp,
    None,
    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# Display result
cv.imshow('sift', gray)

cv.waitKey(0)
cv.destroyAllWindows()