import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import SubtractDominantMotion
import SubtractInverse
import time

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--tolerance', type=float, default=0.2, help='binary threshold of intensity difference when computing the mask')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance
start_time = time.time()
seq = np.load('../data/antseq.npy')
_, _, frame = seq.shape
for i in range(frame - 1):
    print(f'Frame ',(i+1), 'of ', (frame))

    #Get Frames for Analysis
    template_frame = seq[:, :, i]
    curr_frame = seq[:, :, i + 1]

    #Get Image for Mask
    mask = SubtractDominantMotion.SubtractDominantMotion(template_frame, curr_frame, threshold, num_iters, tolerance)
    #mask = SubtractInverse.SubtractInverse(template_frame, curr_frame, threshold, num_iters, tolerance)
    positions_x = []
    positions_y = []
    for j in range(mask.shape[0]):
        for k in range(mask.shape[1]):
            if mask[j, k] == 0:
                positions_y.append(j)
                positions_x.append(k)

    #Create Boxed Image
    if i == 29 or i == 59 or i == 89 or i == 119:
        print("True")
        plt.figure()
        plt.imshow(curr_frame, cmap='gray')
        plt.plot(positions_x, positions_y, ',', markerfacecolor='blue')
        plt.savefig('../results/AntFrame_'+str(i + 1)+'.png', bbox_inches='tight')
        #plt.savefig('../results/InverseAntFrame_'+str(i + 1)+'.png', bbox_inches='tight')
end_time = time.time()
print(f'\nRuntime: {end_time - start_time:.4f} seconds')