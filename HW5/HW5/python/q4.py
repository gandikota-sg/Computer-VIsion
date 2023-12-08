import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation
import matplotlib.pyplot as plt
from PIL import Image
import os

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions

    # Denoise image
    image = skimage.filters.gaussian(image, sigma=2, channel_axis=-1)

    # Greyscale image
    image = skimage.color.rgb2gray(image)

    # Threshold image
    thresh = skimage.filters.threshold_otsu(image)

    # Morphology image
    sqr = skimage.morphology.square(10)
    bw = skimage.morphology.closing(image <= thresh, sqr).astype(np.float32)

    # Label image
    label_img = skimage.measure.label(skimage.segmentation.clear_border(bw))
    image_label_overlay = skimage.color.label2rgb(label_img, image, bg_label=0)

    # Skip small boxes
    for region in skimage.measure.regionprops(label_img):
        if region.area >= 300:
            bboxes.append(region.bbox)

    bw = bw
    return bboxes, bw
