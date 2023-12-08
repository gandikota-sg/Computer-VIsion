# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

# Imports
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from utils import integrateFrankot
import scipy
import cv2

def renderNDotLSphere(center, rad, light, pxSize, res):

    """
    Question 1 (b)

    Render a sphere with a given center and radius. The camera is 
    orthographic and looks towards the sphere in the negative z
    direction. The camera's sensor axes are centerd on and aligned
    with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray
        The center of the hemispherical bowl in an array of size (3,)

    rad : float
        The radius of the bowl

    light : numpy.ndarray
        The direction of incoming light

    pxSize : float
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the hemispherical bowl
    """

    # Calculate center coordinates for camera 
    c_x = res[0] / 2
    c_y = res[1] / 2

    # Find meshgrid and x/y in space
    x, y = np.meshgrid(np.arange(res[0]), np.arange(res[1]))

    x = pxSize * (x - c_x) + center[0]
    y = pxSize * (y - c_y) + center[1]

    # Find z for sphere
    z = rad ** 2 - x ** 2 - y ** 2
    mask = z < 0
    z[mask] = 0.0

    # Compute dot product with light direction
    pts = np.stack((x, y, np.sqrt(z)), axis = 2).reshape((-1, 3))
    pts = pts / np.linalg.norm(pts, axis = 1)[:, np.newaxis]
    image = np.dot(pts, light).reshape(res[::-1])  

    # print(image.shape)
    image[mask] = 0.0
    return image


def loadData(path = "../data/"):

    """
    Question 1 (c)

    Load data from the path given. The images are stored as input_n.tif
    for n = {1...7}. The source lighting directions are stored in
    sources.mat.

    Paramters
    ---------
    path: str
        Path of the data directory

    Returns
    -------
    I : numpy.ndarray
        The 7 x P matrix of vectorized images

    L : numpy.ndarray
        The 3 x 7 matrix of lighting directions

    s: tuple
        Image shape

    """

    I = []
    L = None
    P = 0

    # Iterate through 7 images
    for i in range(7):
        # Generate path for each image
        img_path = path + f"input_{i+1}.tif"
        gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Store the pixel values in the I list
        I.append(gray_img.flatten())
        
        if P == 0:
            P = gray_img.size

    # Convert to array, load, and get dimensions
    I = np.array(I)
    L = np.load(path + "sources.npy").T
    s = gray_img.shape

    return I, L, s


def estimatePseudonormalsCalibrated(I, L):

    """
    Question 1 (e)

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """

    L_new = np.linalg.inv(np.dot(L, L.T))
    B = L_new.dot(L).dot(I)
    return B


def estimateAlbedosNormals(B):

    '''
    Question 1 (e)

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    '''

    albedos = np.linalg.norm(B, axis=0)
    normals = B / (albedos + 0.000001)
    return albedos, normals


def displayAlbedosNormals(albedos, normals, s):

    """
    Question 1 (f)

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `gray` colormap for the albedo image
    and the `rainbow` colormap for the normals.

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """
    albedo = albedos/np.max(albedos)
    albedoIm = np.reshape(albedo, s)

    normal = (normals+1.0) / 2.0
    normalIm = np.reshape(normal.T, (s[0], s[1], 3))

    return albedoIm, normalIm


def estimateShape(normals, s):

    """
    Question 1 (i)

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point

    """

    x, y = (normals[:2] / (-normals[2] + 0.000001)).reshape((2, *s))
    surface = integrateFrankot(x, y)
    return -surface


def plotSurface(surface):

    """
    Question 1 (i) 

    Plot the depth map as a surface

    Parameters
    ----------
    surface : numpy.ndarray
        The depth map to be plotted

    Returns
    -------
        None

    """
    
    fig = plt.figure()
    X, Y = np.meshgrid(np.arange(surface.shape[1]), np.arange(surface.shape[0]))

    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, surface, edgecolor='none', cmap = matplotlib.cm.coolwarm)
    ax.set_title('Surface Plot')

    plt.show()


if __name__ == '__main__':

    # Q1.2
    lights = np.asarray([[1, 1, 1]/np.sqrt(3), [1, -1, 1] /
                         np.sqrt(3), [-1, -1, 1]/np.sqrt(3)])
    for i in range(len(lights)):
        image = renderNDotLSphere(np.asarray([0.0, 0.0, 0.0]), 7.5, lights[i], 7e-3, np.asarray([3840, 2160]))
        cv2.imwrite('../results/q1_b_{}.png'.format(i+1), (image*255))

    # Q1.3 
    I, L, s = loadData()

    # Q1.4 
    print(scipy.linalg.svdvals(I))

    # Q1.5
    albedos, normals = estimateAlbedosNormals(estimatePseudonormalsCalibrated(I, L))

    # Q1.6
    albedo_img, normal_img = displayAlbedosNormals(albedos, normals, s)
    #plt.imshow(albedo_img, cmap='gray')
    #plt.savefig('../results/q1_f_a.png')
    #plt.show()

    #normal_img = (normal_img - np.min(normal_img)) / (np.max(normal_img) - np.min(normal_img))
    #plt.imshow(normal_img, cmap='rainbow')
    #plt.savefig('../results/q1_f_n.png')
    #plt.show()

    # Q1.9
    surface = estimateShape(normals, s)
    surface = (surface - np.min(surface)) / (np.max(surface) - np.min(surface))
    plotSurface(surface)
