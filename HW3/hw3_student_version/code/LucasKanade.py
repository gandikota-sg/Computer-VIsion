import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """
	
    # Put your implementation here
    p = p0
    magnitude = 1
    count = 1

    #Get Image Shapes
    template_frame_height, template_frame_width = It.shape
    init_frame_height, init_frame_width = It1.shape

    #Set Up Rect Info
    x_1, y_1, x_2, y_2 = rect
    rect_height = int(x_2 - x_1)
    rect_width = int(y_2 - y_1)

    #Calculate Splines
    template_x = np.linspace(0, template_frame_height, template_frame_height, False)
    template_y = np.linspace(0, template_frame_width, template_frame_width, False)
    template_spline = RectBivariateSpline(template_x, template_y, It)

    init_x = np.linspace(0, init_frame_height, init_frame_height, False)
    init_y = np.linspace(0, init_frame_width, init_frame_width, False)
    init_spline = RectBivariateSpline(init_x, init_y, It1)

    # Creating a 2D grid of coordinates (x, y)
    complex_num = 1j
    x, y = np.mgrid[x_1:(x_2+1):(rect_width * complex_num), y_1:(y_2+1):(rect_height * complex_num)]

    while (magnitude > threshold) and (count < num_iters):
        #Partial Dirivatives
        movement_x = x + p[0]
        movement_y = y + p[1]
        dp_x = init_spline.ev(movement_y, movement_x, dy = 1).flatten()
        dp_y = init_spline.ev(movement_y, movement_x, dx = 1).flatten()

        It1_p = init_spline.ev(movement_y, movement_x).flatten()
        It_p = template_spline.ev(y, x).flatten()

        #Make Matrix A
        A = np.zeros((rect_width * rect_height, 2 * rect_width * rect_height))
        for i in range(rect_width * rect_height):
            A[i, (2 * i)] = dp_x[i]
            A[i, (2 * i + 1)] = dp_y[i]
        A = np.dot(A, np.tile(np.eye(2), (rect_width * rect_height, 1)))

        #Make Matrix b
        b = np.reshape(It_p - It1_p, (rect_width * rect_height, 1))

        #Pseudoinverse Solve and Update
        magnitude = np.linalg.norm(np.linalg.pinv(A).dot(b))

        #New p Update
        p = (p + np.linalg.pinv(A).dot(b).T).ravel()
        count += 1

    return p
