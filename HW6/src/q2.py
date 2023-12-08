# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

import numpy as np
from q1 import loadData, estimateAlbedosNormals, displayAlbedosNormals
from q1 import estimateShape, plotSurface 
from utils import enforceIntegrability
from matplotlib import pyplot as plt

def estimatePseudonormalsUncalibrated(I):

    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions. 

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals

    """
    u, _, v = np.linalg.svd(I, full_matrices=False)
    B = v[0:3, :]
    L = u[0:3, :]

    return B, L


if __name__ == "__main__":

    # Q2.2
    I, L_data, s = loadData()
    B, L_est = estimatePseudonormalsUncalibrated(I)

    albedos, normals = estimateAlbedosNormals(B)
    albedo_img, normal_img = displayAlbedosNormals(albedos, normals, s)
    
    #plt.imshow(albedo_img, cmap='gray')
    #plt.savefig('../results/q2_b_a.png')
    #plt.show()

    normal_img = (normal_img - np.min(normal_img)) / (np.max(normal_img) - np.min(normal_img))
    #plt.imshow(normal_img, cmap='rainbow')
    #plt.savefig('../results/q2_b_n.png')
    #plt.show()

    print(L_data)
    print(L_est)

    # Q2.4
    surface = estimateShape(normals, s)
    surface = (surface - np.min(surface)) / (np.max(surface) - np.min(surface))
    #plotSurface(surface)

    # Q2.5
    albedos, normals = estimateAlbedosNormals(B)
    normals = enforceIntegrability(normals, s)
    surface = estimateShape(normals, s)
    surface = (surface - np.min(surface)) / (np.max(surface) - np.min(surface))
    #plotSurface(-surface)

    # Q2.6
    mu = 1  
    nu = 2 
    lam = 3 
    G = np.asarray([[1, 0, 0], [0, 1, 0], [mu, nu, lam]])
    B = np.linalg.inv(G.T).dot(B)
    albedos, normals = estimateAlbedosNormals(B)
    normals = enforceIntegrability(normals, s)
    surface = estimateShape(normals, s)
    surface = (surface - np.min(surface)) / (np.max(surface) - np.min(surface))
    plotSurface(-surface)