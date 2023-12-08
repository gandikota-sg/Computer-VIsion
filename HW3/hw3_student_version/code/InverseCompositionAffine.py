import numpy as np
from scipy.interpolate import RectBivariateSpline
import scipy

def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array]
    """

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    Hessian = np.zeros((6, 6))
    p = np.zeros(6)
    error = 1
    count = 1

    #Get Gradients and Calculate Hessian
    x_grad, y_grad = np.gradient(It)
    for i in range(It.shape[0]):
        for j in range(It.shape[1]):
            A_matrix = np.dot(np.array([x_grad[i, j], y_grad[i, j]]), np.array([[j, 0, i, 0, 1, 0], [0, j, 0, i, 0, 1]]))[np.newaxis, :]
            Hessian = Hessian + np.dot(A_matrix.T, A_matrix)
    print("Hessian Found")

    while error > threshold and count < num_iters:
        print("Iteration: ", count)
        count += 1
        b_matrix = np.zeros((6, 1))
        warped_curr_frame = scipy.ndimage.affine_transform(It1, M)

        #Compute gradient of b 
        for i in range(It.shape[0]):
            for j in range(It.shape[1]):
                A_matrix = np.dot(np.array([x_grad[i, j], y_grad[i, j]]), np.array([[j, 0, i, 0, 1, 0], [0, j, 0, i, 0, 1]]))[np.newaxis, :]
                Hessian = Hessian + np.dot(A_matrix.T, A_matrix)
                b_matrix = b_matrix + (np.transpose(A_matrix) * (warped_curr_frame - It))

        #Pseudoinverse Solve and Update
        magnitude = np.dot(np.linalg.pinv(Hessian), (b_matrix))

        #Update M matrix using derivative
        mag_derivative = [
            [1.0 + magnitude[0][0], magnitude[2][0], magnitude[4][0]],
            [magnitude[1][0], 1.0 + magnitude[3][0], magnitude[5][0]],
            [0, 0, 1]]
        M = np.dot(np.concatenate((M, np.array([[0, 0, 1]])), axis=0), np.linalg.pinv(mag_derivative))
        M = M[0:2, :]

        #New p Update
        p = (p + magnitude.T).ravel()
        error = np.linalg.norm(magnitude)

    return M
