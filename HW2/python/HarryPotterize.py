import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts

#Import necessary functions
import matplotlib.pyplot as plt
from matchPics import matchPics
from planarH import computeH_ransac
from planarH import compositeH

#Write script for Q2.2.4
opts = get_opts()

#Reads Cover, Desk, and Harry Potter Cover
cv_desk = cv2.imread('../data/cv_desk.png')
cv_cover = cv2.imread('../data/cv_cover.jpg')
hp_cover = cv2.imread('../data/hp_cover.jpg')

#Compute Homography
matches, locs1, locs2 = matchPics(cv_desk, cv_cover, opts)
best_H2to1, inliers = computeH_ransac(locs2[matches[:,1]], locs1[matches[:,0]], opts)
hp_resize = cv2.resize(hp_cover,(cv_cover.shape[1],cv_cover.shape[0]))

composite_img = compositeH(best_H2to1, hp_resize, cv_desk)
cv2.imwrite('../results/Figure_iter100.jpg', composite_img)

