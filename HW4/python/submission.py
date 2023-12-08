"""
Homework 4
"""
import numpy as np
import util
import helper
import matplotlib.pyplot as plt

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # Get parameter dimensions
    T_matrix = np.array([[(1 / M), 0, 0],
                [0, (1 / M), 0],
                [0, 0, 1]])

    pts1 = np.column_stack((pts1, np.ones(pts1.shape[0])))
    pts2 = np.column_stack((pts2, np.ones(pts1.shape[0])))

    # Scale data and compute A
    pts1_N = np.dot(T_matrix, pts1.T).T
    pts2_N = np.dot(T_matrix, pts2.T).T

    A_matrix = np.column_stack([
        pts1_N[:, 0] * pts2_N[:, 0],
        pts1_N[:, 0] * pts2_N[:, 1],
        pts1_N[:, 0],
        pts1_N[:, 1] * pts2_N[:, 0],
        pts1_N[:, 1] * pts2_N[:, 1],
        pts1_N[:, 1],
        pts2_N[:, 0],
        pts2_N[:, 1],
        np.ones(pts1.shape[0])
    ])

    # Solve using SVD for F
    _, _, V = np.linalg.svd(A_matrix)
    F = util._singularize(np.reshape(V.T[:, -1], (3, 3)).T)
    F = util.refineF(F, pts1_N[:, :2], pts2_N[:, :2])

    # Undo normalization
    F = np.matmul(np.transpose(T_matrix), np.matmul(F, T_matrix))
    F /= F[-1, -1]
    #print(F)
    return F

'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # E matrix calculation E = K2^T * F * K1
    E = np.dot((np.dot(K2.T, F)), K1)
    
    # Normalize E
    E = E / E[-1, -1]
    
    return E


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    # Initialize variables and get point count
    n = pts1.shape[0]
    P = np.zeros((n, 3))
    err = 0

    # Loop for every point to triangulate
    for i in range(n):
        # Get x and y for image 1 and image 2
        x_1, y_1 = pts1[i, 0], pts1[i, 1]
        x_2, y_2 = pts2[i, 0], pts2[i, 1]

        # Make A matrix and SVD
        A_1 = C1[2, :] * x_1 - C1[0, :]
        A_2 = C1[2, :] * y_1 - C1[1, :]
        A_3 = C2[2, :] * x_2 - C2[0, :]
        A_4 = C2[2, :] * y_2 - C2[1, :]
        A = np.vstack((A_1, A_2, A_3, A_4))

        _, _, V = np.linalg.svd(A)
        v = V.T
        res = v[:, -1]

        # Homogenize and project the points
        res_hom = res / res[-1]
 
        im1_2d = np.matmul(C1, res_hom.T)
        im2_2d = np.matmul(C2, res_hom.T)

        # Normalize and get all x, y points
        im1_2d = im1_2d / im1_2d[-1]
        im1_pts = im1_2d[0: 2]
        im2_2d = im2_2d / im2_2d[-1]
        im2_pts = im2_2d[0: 2]

        # Reproject and find triangulation/error
        im1_err = np.linalg.norm(im1_pts - pts1[i, :]) ** 2
        im2_err = np.linalg.norm(im2_pts - pts2[i, :]) ** 2

        P[i, :] = res_hom[0: 3]
        err = err + (im1_err + im2_err)

    return P, err

   

'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Convert input coordinates to integers
    x1 = int(x1)
    y1 = int(y1)
    pt = np.array([x1, y1, 1])
    epipolar_line = np.dot(F, pt)

    # Define the size of the searching window
    size_window = 15
    searching_rect = im1[(y1 - size_window//2): (y1 + size_window//2 + 1),
                      (x1 - size_window//2): (x1 + size_window//2 + 1), :]

    # Normalize the epipolar line
    normal_epi_Line = epipolar_line / np.linalg.norm(epipolar_line)
    # epipolar_y = np.arange(im2.shape[0])
    # epipolar_x = np.rint(-(normal_epi_Line[1] * np.arange(im2.shape[0]) + normal_epi_Line[2]) / normal_epi_Line[0])

    # Define a Gaussian weighting function for error computation
    x_gauss, y_gauss = np.meshgrid(np.arange(-size_window//2, size_window//2+1, 1),
                                 np.arange(-size_window//2, size_window//2+1, 1))
    std_dev = 7
    weight = np.sum(np.dot((np.exp(-((x_gauss**2 + y_gauss**2) / (2 * (std_dev**2))))), 1) / np.sqrt(2*np.pi*std_dev**2))

    # Loop through possible points 
    curr_err = 1e7
    for y2_candidate in range((y1 - size_window//2), (y1 + size_window//2 + 1)):
        x2_candidate = int((-normal_epi_Line[1] * y2_candidate - normal_epi_Line[2]) / normal_epi_Line[0])

        # Check if the current point is within region in the second image
        if (x2_candidate >= size_window//2 and x2_candidate + size_window//2 < im2.shape[1] and
                y2_candidate >= size_window//2 and y2_candidate + size_window//2 < im2.shape[0]):

            # Find corresponding rectangle
            im2_rect = im2[y2_candidate - size_window//2:y2_candidate + size_window//2 + 1,
                           x2_candidate - size_window//2:x2_candidate + size_window//2 + 1, :]
            err = np.linalg.norm((searching_rect - im2_rect) * weight)

            # Update error and point
            if err < curr_err:
                curr_err = err
                x2 = x2_candidate
                y2 = y2_candidate

    return x2, y2



'''
Q5.1: Extra Credit RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M, nIters=500, tol=10):
    # Initialize necessary variables
    num_samples = pts1.shape[0]
    max_iters = nIters
    inliers = np.zeros((num_samples, 1))

    # Iterate for all iterations
    for i in range(nIters):
        # Get random points for eighpoint
        random_indexes = np.random.choice(num_samples, size=8)
        rand_pts1, rand_pts2 = pts1[random_indexes, :], pts2[random_indexes, :]

        F = eightpoint(rand_pts1, rand_pts2, M)

        # Find inliers and which ones are acceptable
        pts1_all, pts2_all = np.hstack((pts1, np.ones((num_samples, 1)))), np.hstack((pts2, np.ones((num_samples, 1))))

        # Error calculate using sum or squared distance
        dist_1 = np.square(np.divide(np.sum(np.multiply(
            pts1_all.dot(F.T), pts2_all), axis=1), np.linalg.norm(pts1_all.dot(F.T)[:, :2], axis=1)))
        dist_2 = np.square(np.divide(np.sum(np.multiply(
             pts2_all.dot(F.T), pts1_all), axis=1), np.linalg.norm(pts2_all.dot(F.T)[:, :2], axis=1)))
        err = (dist_1 + dist_2).flatten()

        # Validate inliers
        valid_inlier = (err < tol).astype(np.int8)
        if valid_inlier[valid_inlier == 1].shape[0] > inliers[inliers == 1].shape[0]:
            inliers = valid_inlier

    pts1_inlier = pts1[np.where(inliers == 1)]
    pts2_inlier = pts2[np.where(inliers == 1)]
    F = eightpoint(pts1_inlier, pts2_inlier, M)

    return F, inliers


'''
Q5.2:Extra Credit  Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # https://courses.cs.duke.edu/cps274/fall13/notes/rodrigues.pdf
    # https://people.eecs.berkeley.edu/~ug/slide/pipeline/assignments/as5/rotation.html

    # Initialize all required terms
    I = np.eye(3)
    axis_r = np.concatenate(r).T

    # Find angle of rotation and unit vector
    angle_r = np.linalg.norm(axis_r)
    unit_vector_r = axis_r / angle_r

    # Calculate the cross matrix
    cross_product_matrix = np.array([[0, -unit_vector_r[2], unit_vector_r[1]],
                                    [unit_vector_r[2], 0, -unit_vector_r[0]],
                                    [-unit_vector_r[1], unit_vector_r[0], 0]])

    # Calculate Rodrigues rotation matrix
    R = I * np.cos(angle_r) + \
                      (1 - np.cos(angle_r)) * np.outer(unit_vector_r, unit_vector_r) + \
                      cross_product_matrix * np.sin(angle_r)

    return R

'''
Q5.2:Extra Credit  Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # https://courses.cs.duke.edu/cps274/fall13/notes/rodrigues.pdf
    # https://people.eecs.berkeley.edu/~ug/slide/pipeline/assignments/as5/rotation.html

    # Caulcate matrix A and point vector
    skew_mat = (R - R.T) / 2
    p_vec = np.array([skew_mat[2,1], skew_mat[0, 2], skew_mat[1, 0]]).T

    # Calculate rotation angle and axis
    axis_r = p_vec/5
    angle_r = np.arctan2(np.linalg.norm(p_vec), (R[0, 0] + R[1, 1] + R[2, 2] - 1) / 2)

    #Calculate rotation and scaled axis
    r = axis_r * angle_r
    return r

'''
Q5.3: Extra Credit Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    pass

'''
Q5.3 Extra Credit  Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    pass

if __name__ == "__main__":

    # Loading correspondences
    correspondence = np.load('../data/some_corresp_noisy.npz')
    
    # Loading the intrinscis of the camera
    intrinsics = np.load('../data/intrinsics.npz')
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')
    M = np.max([*im1.shape, *im2.shape])

    F_orig = eightpoint(pts1, pts2, M)
    np.savez('../data/q2_1', F_orig)
    nIters = 500  
    tol = 10
    F_ransac, inliers = ransacF(pts1, pts2, M, nIters, tol)
    
    # helper.displayEpipolarF(im1, im2, F_orig)
    helper.displayEpipolarF(im1, im2, F_ransac)

    '''E = essentialMatrix(F, K1, K2)
    print(E)
    np.savez('../data/q3_1', E)

    #helper.displayEpipolarF(im1, im2, F)
    x2, y2 = epipolarCorrespondence(im1, im2, F, 119, 217)
    helper.epipolarMatchGUI(im1, im2, F)'''
