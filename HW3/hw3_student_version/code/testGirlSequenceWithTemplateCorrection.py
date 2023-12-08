import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import LucasKanade

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--template_threshold', type=float, default=5, help='threshold for determining whether to update template')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold
    
seq = np.load("../data/girlseq.npy")
rect = [280, 152, 330, 318]

#Set up array for all boxes
_, _, frame = seq.shape
print(seq.shape)
box_array = rect.copy()
template_array = rect.copy()
template_frame = seq[:, :, 0]
move_vec_orig = np.zeros(2)

for i in range(frame - 1):
    print(f'Frame ',(i+1), 'of ', (frame))
    #Get frames for analysis
    curr_frame = seq[:, :, i+1]

    #Calculate Lucas Kanade Update with Drift Correction
    movement_vec = LucasKanade.LucasKanade(template_frame, curr_frame, template_array, threshold, num_iters, move_vec_orig)
    adjusted_vec = movement_vec + [template_array[0] - rect[0], template_array[1]-rect[1]]
    updated_vec = LucasKanade.LucasKanade(seq[:, :, 0], curr_frame, rect, threshold, num_iters, adjusted_vec)
    delta_movement = np.linalg.norm(adjusted_vec - updated_vec)
    #print(delta_movement)

    if delta_movement < template_threshold:
        change_vec = (updated_vec - [template_array[0]-rect[0], template_array[1] - rect[1]])
        for j in range(4):
            template_array[j] += change_vec[j % 2]
        template_frame = seq[:, :, i+1]
        box_array = np.vstack((box_array, template_array))
        move_vec_orig = np.zeros(2)
    else:
        box_array = np.vstack(
            (box_array, [template_array[0]+movement_vec[0], template_array[1]+movement_vec[1], template_array[2]+movement_vec[0], template_array[3]+movement_vec[1]]))
        move_vec_orig = movement_vec
    #print(box_array)

    #Create Boxed Image
    girlseq = np.load('girlseqrects.npy')
    if i == 0 or i == 19 or i == 39 or i == 59 or i == 79:
        print("True")
        plt.figure()
        plt.imshow(curr_frame, cmap='gray')
        no_update = girlseq[i + 1, :]
        updated = box_array[-1, :]
        patch_no_update = patches.Rectangle((no_update[0], no_update[1]), (no_update[2] - no_update[0]),
                                    (no_update[3] - no_update[1]), edgecolor='b', linewidth=3, facecolor='none')
        patch_update = patches.Rectangle((updated[0], updated[1]), (updated[2] - updated[0]),
                                    (updated[3] - updated[1]), edgecolor='r', linewidth=3, facecolor='none')
        ax = plt.gca()
        ax.add_patch(patch_no_update)
        ax.add_patch(patch_update)
        plt.savefig('../results/GirlCorrectionFrame_' + str(i + 1)+'.png', bbox_inches='tight')

np.save('girlseqrects-wcrt.npy', box_array)
