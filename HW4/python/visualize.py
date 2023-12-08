import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import submission
import helper

'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''

# Import all necesary files
temple_coords = np.load('../data/templeCoords.npz')

corresp = np.load('../data/some_corresp.npz')
pts1, pts2 = corresp['pts1'], corresp['pts2']

intrinsics = np.load('../data/intrinsics.npz')
K1, K2 = intrinsics['K1'], intrinsics['K2']

im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')

# Calculate and plot 3D points 
F = submission.eightpoint(pts1, pts2, np.max([*im1.shape, *im2.shape]))
t_pts1 = np.hstack((temple_coords["x1"], temple_coords["y1"]))
t_pts2 = np.hstack((temple_coords["x1"], temple_coords["y1"]))

# Loop to find for all points
for i in range(temple_coords["x1"].shape[0]):
    # Get current necessary points
    x_1 = temple_coords["x1"][i]
    y_1 = temple_coords["y1"][i]

    # Calculate the epipolar correspondence
    t_pts2[i, 0], t_pts2[i, 1] = submission.epipolarCorrespondence(im1, im2, F, x_1, y_1)

# Find M2 and save results
M1 = np.hstack((np.eye(3), np.zeros((3,1))))
E = submission.essentialMatrix(F, K1, K2)
M2 = helper.camera2(E)
C1 = np.matmul(K1, M1)

cur_err = np.inf
# Loop to find best error for P
for i in range(M2.shape[2]):
	C2 = np.dot(K2, M2[:, :, i])
	P, err = submission.triangulate(C1, t_pts1, C2, t_pts2)

	if err < cur_err and np.min(P[:, 2]) >= 0:
		cur_err = err
		M2_final = M2[:, :, i]
		C2_final = C2
		P_final = P

np.savez('q4_2.npz', F,  M1,  M2_final,  C1, C2_final)

# Plot results
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter3D(P_final[:, 0], P_final[:, 1], P_final[:, 2])
plt.show()
