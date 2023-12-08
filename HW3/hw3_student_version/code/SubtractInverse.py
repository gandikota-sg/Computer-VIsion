import numpy as np
import InverseCompositionAffine
import scipy

def SubtractInverse(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """
    
    # put your implementation here
    mask = np.ones(image1.shape, dtype=bool)

    #Find Matrix M and Warp Image
    matrix_M = InverseCompositionAffine.InverseCompositionAffine(image1, image2, threshold, num_iters)
    warped_image = scipy.ndimage.affine_transform(image1, -matrix_M, offset=0.0, output_shape=None)

    #Generate Image Mask
    mask = (np.abs(warped_image - image2) > tolerance)
    mask = scipy.ndimage.morphology.binary_erosion(mask)
    mask = scipy.ndimage.morphology.binary_dilation(mask)
    print("Mask Made")

    return mask
