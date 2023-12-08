import numpy as np
import cv2
from matchPics import matchPics
from opts import get_opts
import scipy
import matplotlib.pyplot as plt

opts = get_opts()
#Q2.1.6
#Read the image and convert to grayscale, if necessary
cv_cover = cv2.imread('../data/cv_cover.jpg')
matches_arr = []

for i in range(36):
	#Rotate Image
	angle = i * 10
	#print(angle)
	cover_rotated = scipy.ndimage.rotate(cv_cover, angle, reshape=False)

	#Compute features, descriptors and Match features
	matches, locs1, locs2 = matchPics(cv_cover, cover_rotated, opts)

	#Update histogram
	matches_arr.append(len(matches))

	#pass # comment out when code is ready


#Display histogram
plt.figure(1)
plt.bar(np.arange(0,360,10), matches_arr, alpha=0.75, width=7)
plt.xlabel('Rotations in degrees')
plt.ylabel('Number of Matches')
plt.title('Number of Matches for Every 10 Angles')
plt.show()
