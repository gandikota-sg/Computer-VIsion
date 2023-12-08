import os, multiprocessing
from os.path import join, isfile

import numpy as np
from PIL import Image
import scipy.ndimage
import skimage.color
import sklearn.cluster
from opts import get_opts
from multiprocessing import Pool

def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    filter_scales = opts.filter_scales

    #Check Gray Scale or not 3 Channel
    if len(img.shape) < 3:
        img = np.tile(img[:, :, np.newaxis], (1, 1, 3))
    if img.shape[2] > 3:
        img = img[:, :, :3]

    #Check if floating point type
    #if img.dtype != np.float32 or np.min(img) < 0 or np.max(img) > 1:
    #    img = img.astype(np.float32) / 255.0

    #Convert RGB to LAB color space
    img_lab = skimage.color.rgb2lab(img)

    #Check all filters
    filter_responses = []
    for s in filter_scales:
        # Gaussian filter
        for i in range(0, 3): 
            filter_responses.append(scipy.ndimage.gaussian_filter(img_lab[:, :, i], sigma=s))
        # Laplacian of gaussian filter
        for i in range(0, 3): 
            filter_responses.append(scipy.ndimage.gaussian_laplace(img_lab[:, :, i], sigma=s))
        #Derivative of gaussian in X axis
        for i in range(0, 3): 
            filter_responses.append(scipy.ndimage.gaussian_filter(img_lab[:, :, i], sigma=s, order=[1, 0]))
        #Derivative of gaussian in Y axis 
        for i in range(0, 3): 
            filter_responses.append(scipy.ndimage.gaussian_filter(img_lab[:, :, i], sigma=s, order=[0, 1]))
    filter_responses = np.dstack(filter_responses)
    return filter_responses

def compute_dictionary_one_image(args):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    '''
    opts = get_opts()

    #Split args into corresponding variables
    train_files = args[0]
    img_num = args[1]
    alpha = int(args[2])

    #Get image data, help from Q1.1
    img_path = join(opts.data_dir, train_files)
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32) / 255

    #Responses at alpha random pixels
    filter_response = extract_filter_responses(opts, img)
    #https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.choice.html
    new_img = filter_response[np.random.choice(filter_response.shape[0], alpha), np.random.choice(filter_response.shape[1], alpha)]
    img_path = join(opts.feat_dir, str(img_num) + '.npy')
    np.save(img_path, new_img)

    

def compute_dictionary(opts, n_worker):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel
    
    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    
    #Get alpha and image list
    img_list = np.arange(len(train_files))
    alpha_list = np.ones(len(train_files)) * opts.alpha

    #Attempt Multiprocessing https://docs.python.org/3/library/multiprocessing.html
    multi_process = Pool(n_worker)
    args = []
    for t, i, a in zip(train_files, img_list, alpha_list):
        args.append((t, i, a))
    multi_process.map(compute_dictionary_one_image, args)

    #Get image filer responses
    filter_responses = []
    print(len(train_files))
    for i in range(len(train_files)):
        img_path = join(opts.feat_dir, str(i) + '.npy')
        filter_responses.append(np.load(img_path))

    #Implement KMeans
    kmeans = sklearn.cluster.KMeans(n_clusters=K).fit(np.concatenate(filter_responses, axis=0))
    dictionary = kmeans.cluster_centers_
    np.save(join(out_dir, 'dictionary.npy'), dictionary)

def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''

    #Filter Image
    resp = extract_filter_responses(opts, img)
    resp = resp.reshape(img.shape[0] * img.shape[1], dictionary.shape[-1])
    
    #Assign Closest Word
    euc_dist = scipy.spatial.distance.cdist(resp, dictionary, 'euclidean')
    words = np.argmin(euc_dist, axis = 1)
    wordmap = words.reshape(img.shape[0], img.shape[1])
    return wordmap
