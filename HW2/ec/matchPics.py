import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection

def matchPics(I1, I2, opts):
	#I1, I2 : Images to match
	#opts: input opts
	ratio = opts.ratio  #'ratio for BRIEF feature descriptor'
	sigma = opts.sigma  #'threshold for corner detection using FAST feature detector'
	
	#Convert Images to GrayScale
	#https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html 
	I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
	I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
	
	#Detect Features in Both Images
	features_1 = corner_detection(I1, opts.sigma)
	features_2 = corner_detection(I2, opts.sigma)

	#Obtain descriptors for the computed feature locations
	descrip_1, locs1 = computeBrief(I1, features_1)
	descrip_2, locs2 = computeBrief(I2, features_2)

	#Match features using the descriptors
	matches = briefMatch(descrip_1, descrip_2, opts.ratio)

	return matches, locs1, locs2
