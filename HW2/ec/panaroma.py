import numpy as np
import cv2

#Import necessary functions
from opts import get_opts
from matchPics import matchPics
from planarH import computeH_ransac, compositeH
import matplotlib.pyplot as plt
from helper import plotMatches

# Write script for Q4.2x
#Get Images to Pano
img_left = cv2.imread('../data/online_left.jpg')
img_right = cv2.imread('../data/online_right.jpg')

#Get Image Dimensions
img_left_height, img_left_weight, _ = img_left.shape
img_right_height, img_right_weight, _ = img_right.shape

# Calculate the width for image adjustment
max_weight = max(img_left_weight, img_right_weight)
adjusted_width = round(max_weight * 1.2)

# Create a padded version of img_right to match img_left dimensions
padded_img_right = cv2.copyMakeBorder(
    img_right,
    top=0,
    bottom=img_right_height - img_left_height,
    left=adjusted_width - img_right_weight,
    right=0,
    borderType=cv2.BORDER_CONSTANT,
    value=0
)

# Match features between img_left and padded_img_right
matches, locs1, locs2 = matchPics(img_left, padded_img_right, get_opts())

# Display the matched features
plotMatches(img_left, padded_img_right, matches, locs1, locs2)


#Find the Matched Points
locs1 = locs1[matches[:, 0], 0:2]
locs2 = locs2[matches[:, 1], 0:2]

#Compute Homography
bestH2to1, inliers = computeH_ransac(locs1, locs2, get_opts())
img_pano = compositeH(bestH2to1, img_left, padded_img_right)

#Max and Generate Panoramic
pano_im = np.maximum(padded_img_right, img_pano)
print('Done')
cv2.imwrite('../results/online_pano.png', img_pano)
