'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
import numpy as np
import matplotlib.pyplot as plt
import submission
import helper

# Get images and other imports
im_1 = plt.imread('../data/im1.png')
im_2 = plt.imread('../data/im2.png')
pts = np.load('../data/some_corresp.npz')
K_matrix = np.load('../data/intrinsics.npz')

# Get points for images and corresponsdances
pts_1 = pts['pts1']
pts_2 = pts['pts2']

# Get camera matrix info
K1 = K_matrix['K1']
K2 = K_matrix['K2']

# Compute F and E matrix
F = submission.eightpoint(pts_1, pts_2, np.max(im_1.shape))
E = submission.essentialMatrix(F, K1, K2)

# Calculate M1 and M2
M_1 = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0]])
M_2s = helper.camera2(E)
C1 = np.matmul(K1, M_1)
err = np.inf

# Find the camera matrices and save the best one
best_M2 = 0
for i in range(4):
    curr_M2 = M_2s[:, :, i]
    C2 = np.dot(K2, curr_M2)
    P, curr_err = submission.triangulate(C1, pts_1, C2, pts_2)
    #print(curr_err)
    if err < curr_err:
        best_M2 = i
        err = curr_err

M2 = M_2s[:, :, best_M2]
C1 = np.dot(K1, M_1)
C2 = np.dot(K2, M2)
P, err = submission.triangulate(C1, pts_1, C2, pts_2)

np.savez('../data/q3_3.npz', M2, C2, P)
