import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array] put your implementation here
    """
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    p = np.zeros(6)
    error = 1
    count = 1

    #Get Image Shapes
    template_frame_height, template_frame_width = It.shape
    init_frame_height, init_frame_width = It1.shape

    #Calculate Splines
    template_x = np.linspace(0, template_frame_height, template_frame_height, False)
    template_y = np.linspace(0, template_frame_width, template_frame_width, False)
    template_spline = RectBivariateSpline(template_x, template_y, It)

    init_x = np.linspace(0, init_frame_height, init_frame_height, False)
    init_y = np.linspace(0, init_frame_width, init_frame_width, False)
    init_spline = RectBivariateSpline(init_x, init_y, It1)

    #Creating a mesh grid of coordinates (x, y)
    grid_x, grid_y = np.meshgrid(range(template_frame_width), range(template_frame_height))
    x = grid_x.flatten()
    y = grid_y.flatten()
    coords = np.vstack((x, y, np.ones(template_frame_height * template_frame_width)))

    while error > threshold and count < num_iters:
        count += 1
        #Update transformation matrix M with p
        M[0, 0] = 1 + p[0]
        M[0, 1] = p[1]
        M[0, 2] = p[2]
        M[1, 0] = p[3]
        M[1, 1] = p[4] + 1
        M[1, 2] = p[5]

        #Tranform and remove out of bounds coordinates
        warped_x = np.dot(M, coords)[0]
        warped_y = np.dot(M, coords)[1]
        bad_coords = find_bad_coordinates(warped_x, warped_y, template_frame_width, template_frame_height)
        valid_coords = np.logical_not(bad_coords)
        x_new = x[valid_coords]
        x_new = x_new[:, np.newaxis]
        y_new = y[valid_coords]
        y_new = y_new[:, np.newaxis]
        warped_x = warped_x[valid_coords]
        warped_x = warped_x[:, np.newaxis]
        warped_y = warped_y[valid_coords]
        warped_y = warped_y[:, np.newaxis]

        #Find partial derivatives
        dp_x = init_spline.ev(warped_y, warped_x, dy = 1).flatten()
        dp_x = dp_x[:, np.newaxis]
        dp_y = init_spline.ev(warped_y, warped_x, dx = 1).flatten()
        dp_y = dp_y[:, np.newaxis]
        It_p = template_spline.ev(y_new.flatten(), x_new.flatten()).flatten()
        It1_p = init_spline.ev(warped_y.flatten(), warped_x.flatten()).flatten()
        
        #Calculate matrices
        A = np.hstack((x_new * dp_x, y_new * dp_x, dp_x, x_new * dp_y, y_new * dp_y, dp_y))
        b = (It_p - It1_p).reshape(-1, 1)

        #Pseudoinverse Solve and Update
        magnitude = np.linalg.norm(np.linalg.pinv(A).dot(b))
        p = (p + np.linalg.pinv(A).dot(b).T).ravel()

    #Update transformation matrix M with p
    M[0, 0] = 1 + p[0]
    M[0, 1] = p[1]
    M[0, 2] = p[2]
    M[1, 0] = p[3]
    M[1, 1] = p[4] + 1
    M[1, 2] = p[5]
    return M

def find_bad_coordinates(x, y, width, height):
    x_valid = np.logical_or(x >= width, x < 0)
    y_valid = np.logical_or(y >= height, y < 0)
    if not np.any(x_valid) and not np.any(y_valid):
        bad_coord = []
    elif np.any(x_valid) and not np.any(y_valid):
        bad_coord = np.where(x_valid)[0]
    elif not np.any(x_valid) and np.any(y_valid):
        bad_coord = np.where(y_valid)[0]
    else:
        bad_coord = np.unique(np.concatenate((np.where(x_valid)[0], np.where(y_valid)[0])))
    return bad_coord
