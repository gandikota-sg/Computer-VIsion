import numpy as np
import cv2

#Import necessary functions
import skimage.io 
import skimage.color
from opts import get_opts
from matchPics import matchPics
from planarH import computeH_ransac
from planarH import compositeH
from loadVid import loadVid
from PIL import Image

#Write script for Q3.1
opts = get_opts()

#Load in video and image data
ar_vid = loadVid('../data/ar_source.mov')
cv_cover = cv2.imread('../data/cv_cover.jpg')
book_vid = loadVid('../data/book.mov')

#Set up feature matching
locs1_arr=[]
locs2_arr=[]
matches_arr=[]
bestH2to1_arr=[]
composite_list=[]

#Create video padding
frame_difference = book_vid.shape[0] - ar_vid.shape[0]
frames_to_pad = []
for i in range(frame_difference):
    frames_to_pad.append(ar_vid[i, :, :, :])
video_pad = np.stack(frames_to_pad, axis=0)
ar_vid = np.concatenate((ar_vid, video_pad), axis=0)

#Start feature mapping and homography
for i in range(book_vid.shape[0]):
    matches, locs1, locs2 = matchPics(book_vid[i,:,:,:], cv_cover, opts)
    locs1[:, [0,1]] = locs1[:, [1,0]]
    locs2[:, [0,1]] = locs2[:, [1,0]]
    bestH2to1, inliers = computeH_ransac(locs1[matches[:,0]], locs2[matches[:,1]], opts)
    dimension =(cv_cover.shape[1], cv_cover.shape[0])
    aspect_ratio = cv_cover.shape[1] / cv_cover.shape[0]

    #Create video cover
    new_cover = ar_vid[i,:,:,:]
    
    #Remove black bars
    new_cover = new_cover[44:-44,:]
    height, weight, channels = new_cover.shape
    width_ar = height * cv_cover.shape[1] / cv_cover.shape[0]
    new_cover = new_cover[:, int((weight/2) - width_ar/2):int((weight/2) + width_ar/2)]
    new_cover = cv2.resize(new_cover, dimension)
    
    # Combine augment into frame
    composite_img = compositeH(bestH2to1, new_cover, book_vid[i,:,:,:])
    composite_list.append(composite_img) 
    print(str(i) + " out of " + str(book_vid.shape[0]))
	
#Generate video
#https://www.geeksforgeeks.org/saving-a-video-using-opencv/#
fps = 25
output_dir = '../results/ar.avi'
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
video = None
size = composite_list[0].shape[1], composite_list[0].shape[0]

for img in composite_list:
    if video is None:
        video = cv2.VideoWriter(output_dir, fourcc, float(fps), size, True)
    video.write(np.uint8(img))

video.release()

