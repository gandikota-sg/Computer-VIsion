import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

def sort_by_rightmost_coordinate(boxes, val):
        return sorted(boxes, key=lambda box: box[val])

count = 0
for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    count += 1
    print(f"Image: {count}")
    bboxes, bw = findLetters(im1)

    plt.imshow(bw, cmap = "Greys")
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    #plt.show()

    # find the rows using..RANSAC, counting, clustering, etc.
    # Initialize required variables
    row_boxes = []  
    curr_row = []  
    row = 1  

    # Sort bounding boxes by their right coordinate
    sorted_bboxes = sort_by_rightmost_coordinate(bboxes, 2)
    bottom_coord = bboxes[0][2]

    # Group bounding boxes by rows based on their position
    for box in bboxes:
        top, left, bottom, right = box  
        if top >= bottom_coord:  
            bottom_coord = bottom  
            row_boxes.append(curr_row)  
            curr_row = []  
            row += 1  
        curr_row.append(box)  
    row_boxes.append(curr_row)
    
    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))
   
    for row in row_boxes:
        txt = ""
        # Sort boxes in the row by their rightmost coordinate
        sorted_bboxes = sort_by_rightmost_coordinate(row, 1)
        prev = row[0][3]

        # Iterate through each box in the row
        for box in row:
            top, left, bottom, right = box
            prev = right

            letter = bw[top:bottom, left:right]
            letter = np.pad(letter, (30, 30), 'constant', constant_values=(1, 1))
            letter = skimage.transform.resize(letter, (32, 32))
            letter = letter.T

            h1 = forward(letter.reshape(1, 32*32), params, 'layer1')
            probs = forward(h1, params, 'output', softmax)
            txt = txt + letters[np.argmax(probs[0, :])]
        print(txt)
    
    