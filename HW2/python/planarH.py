import numpy as np
import cv2

def computeH(x1, x2):
	#Q2.2.1
	#Compute the homography between two sets of points
	N = x1.shape[0]
	A = np.zeros([2 * N, 9])

	for i in range(N):
		A[2*i,   :] = [-x2[i, 0], -x2[i, 1], -1, 0, 0, 0, x2[i, 0]*x1[i, 0], x2[i, 1]*x1[i, 0], x1[i, 0]]
		A[2*i+1, :] = [0, 0, 0, -x2[i, 0], -x2[i, 1], -1, x2[i, 0]*x1[i, 1], x2[i, 1]*x1[i, 1], x1[i, 1]]
    
    # Solve for h (homography matrix in flattened form)
	_, _, V = np.linalg.svd(A)
	h = V[-1, :] / V[-1, -1]
    
    # Reshape h to get the homography matrix
	H2to1 = h.reshape((3, 3))
	return H2to1


def computeH_norm(x1, x2):
	#Q2.2.2
	#Compute the centroid of the points
	mean_x1 = np.mean(x1[:, 0])
	mean_y1 = np.mean(x1[:, 1])

	mean_x2 = np.mean(x2[:, 0])
	mean_y2 = np.mean(x2[:, 1])

	x1_centroid = [mean_x1, mean_y1]
	x2_centroid = [mean_x2, mean_y2]

	#Shift the origin of the points to the centroid
	x1_shifted = x1 - x1_centroid
	x2_shifted = x2 - x2_centroid

	#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
	max_distance_x1 = np.max(np.sqrt((x1_shifted[:, 0] ** 2) + (x1_shifted[:, 1] ** 2)))
	max_distance_x2 = np.max(np.sqrt((x2_shifted[:, 0] ** 2) + (x2_shifted[:, 1] ** 2)))
	scale_x1 = np.sqrt(2) / max_distance_x1
	scale_x2 = np.sqrt(2) / max_distance_x2

	#Similarity transform 1
	scale_mat_x1 = np.eye(3) * scale_x1
	translation_mat_x1 = np.array([[1, 0, -mean_x1], [0, 1, -mean_y1], [0, 0, 1]])
	T1 = np.dot(scale_mat_x1, translation_mat_x1)

	# Similarity transform 2
	scale_mat_x2 = np.eye(3) * scale_x2
	translation_mat_x2 = np.array([[1, 0, -mean_x2], [0, 1, -mean_y2], [0, 0, 1]])
	T2 = np.dot(scale_mat_x2, translation_mat_x2)

	#Compute homography
	H_norm = computeH(x1_shifted, x2_shifted)

	#Denormalization
	H2to1 = np.linalg.inv(T1) @ H_norm @ T2

	return H2to1

def computeH_ransac(locs1, locs2, opts):
	#Q2.2.3
	#Compute the best fitting homography given a list of matching points
	max_iters = opts.max_iters  # the number of iterations to run RANSAC for
	inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier
		
	locs1_homogeneous = np.hstack((locs1[:, [1,0]], np.ones((locs1[:, [1,0]].shape[0], 1))))
	locs2_homogeneous = np.hstack((locs2[:, [1,0]], np.ones((locs2[:, [1,0]].shape[0], 1))))
	
	inlier_count = 0
	for i in range(opts.max_iters):
		#Get Random 4 Point Pairs
		np.random.seed(seed = None)
		random_points = np.random.choice(locs1[:, [1,0]].shape[0], 4, True)
		locs1_sample = locs1[:, [1,0]][random_points, :]
		locs2_sample = locs2[:, [1,0]] [random_points, :]
		
		#Compute Homography
		H = computeH_norm(locs1_sample, locs2_sample)     

		#Transform with Homography   
		locs2_trans = H @ locs2_homogeneous.T
		locs2_trans /= locs2_trans[2, :]
		locs2_trans = locs2_trans.T

		#Inliers Calculation
		error_calc = np.linalg.norm(locs1_homogeneous - locs2_trans, axis=1)
		inliers = np.sum(error_calc < opts.inlier_tol)
		
		#Select best fits
		if inlier_count <= inliers:
			inlier_count = inliers
			bestH2to1 = H
	
	return bestH2to1, inliers

def compositeH(H2to1, template, img):
	#Create a composite image after warping the template image on top
	#of the image using the homography

	#Note that the homography we compute is from the image to the template;
	#x_template = H2to1*x_photo

	#Create Image Masks
	img_mask = cv2.warpPerspective(np.ones((template.shape)), np.linalg.inv(H2to1), (img.shape[1], img.shape[0]))
	
	#Warp Template 
	warp_homography = cv2.warpPerspective(template, np.linalg.inv(H2to1), (img.shape[1], img.shape[0]))
	inverted_mask = (img_mask == 0).astype(int)

	#Apply Mask and Form Composite
	composite_img = inverted_mask * img + warp_homography	
	print(composite_img.dtype)
	#Convert Color Scheme
	#Acomposite_img = composite_img.astype(np.float32)
	#composite_img = cv2.cvtColor(composite_img, cv2.COLOR_BGR2RGB)
	return composite_img
	

	


