import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# write your script here, we recommend the above libraries for making your animation
import LucasKanade

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold

seq = np.load("../data/carseq.npy")
rect = [59, 116, 145, 151]

#Set up array for all boxes
_, _, frame = seq.shape
box_array = rect.copy()
for i in range(frame - 1):
    print(f'Frame ',(i+1), 'of ', (frame))

    #Get frames for analysis
    template_frame = seq[:, :, i]
    curr_frame = seq[:, :, i + 1]

    #Calculate Lucas Kanade Update
    movement_vec = LucasKanade.LucasKanade(template_frame, curr_frame, rect, threshold, num_iters)
    #print(movement_vec)
    for j in range(4):
        rect[j] += movement_vec[j % 2]
    box_array = np.vstack((box_array, rect))

    #Create Boxed Image
    if i == 0 or i == 99 or i == 199 or i == 299 or i == 399:
        print("True")
        plt.figure()
        plt.imshow(curr_frame, cmap='gray')
        patch = patches.Rectangle((rect[0], rect[1]), (rect[2] - rect[0]),
                                    (rect[3] - rect[1]), edgecolor='r', linewidth=3, facecolor='none')
        ax = plt.gca()
        ax.add_patch(patch)
        plt.savefig('../results/CarFrame_'+str(i + 1)+'.png', bbox_inches='tight')

np.save('carseqrects.npy', box_array)
