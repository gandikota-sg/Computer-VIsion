import os, math, multiprocessing
from os.path import join
from copy import copy

import numpy as np
from PIL import Image
from multiprocessing import Pool
import visual_words


def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    K = opts.K

    #Get histogram information
    hist_pre, bins = np.histogram(wordmap.flatten(), bins = np.arange(0, K + 1))
    hist = hist_pre / np.sum(hist_pre)

    return hist

def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    '''
        
    K = opts.K
    L = opts.L
    
    #Get dictionary
    dictionary = np.load(join(opts.out_dir, 'dictionary.npy'))
    
    hist_all = np.array([])
    #Pyramid calculations
    for i in range(L, -1, -1):
        l = pow(2, i)
        #Check weights
        if i > 1: #Non Base Layer
            weight = pow(2, i - L - 1)
        else: #Base Layer
            weight = pow(2, -L)
        
        #Creates sub matrices using Wordmap division
        level_mapping = []
        for j in np.array_split(wordmap, l, axis=0):
            for k in np.array_split(j, l, axis=1):
                level_mapping.append(k)

        #Calculate and Append all Histograms to an Array
        for total in range(pow(l, 2)):
            curr_hist = get_feature_from_wordmap(opts, level_mapping[total])
            hist_all = np.append(hist_all, curr_hist, axis = 0)
        
        hist_all = hist_all * weight

    #Normalize Histograms
    if np.sum(hist_all) > 0:
        hist_all = hist_all/np.sum(hist_all)

    return hist_all
    
def get_image_feature(opts, img_path, dictionary):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K*(4^L-1)/3)
    '''

    #Load Image
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255

    #Extract Wordmap
    wordmap = visual_words.get_visual_words(opts, img, dictionary)

    #Compute SPM
    features = get_feature_from_wordmap_SPM(opts, wordmap)
    return features

def build_recognition_system(opts, n_worker=1):
    '''
    Creates a trained recognition system by generating features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy'))

    #Calculate features with multiprocessing map
    features = multiprocess_map_features(n_worker, opts, train_files, dictionary)
    '''features = []

    for f in train_files:
        img_path = join(data_dir, f)
        feature = get_image_feature(opts, img_path, dictionary)
        features.append(feature)
    '''
    features = np.vstack(features)

    # example code snippet to save the learned system
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
        features=features,
        labels=train_labels,
        dictionary=dictionary,
        SPM_layer_num=SPM_layer_num,
    )

def process_image_feature(args):
    img_path, dictionary, opts = args
    feature = get_image_feature(opts, img_path, dictionary)
    return feature

def multiprocess_map_features(n_worker, opts, train_files, dictionary):
    muti_process = Pool(processes=n_worker)
    args_list = [(join(opts.data_dir, f), dictionary, opts) for f in train_files]
    features = muti_process.map(process_image_feature, args_list)
    return features
    
def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * hist_dist: numpy.ndarray of shape (N)
    '''

    #Calculates Similarity and Distance
    similarity = np.sum(np.minimum(word_hist, histograms), axis=1)
    hist_dist = 1 - similarity

    return hist_dist
    
def evaluate_recognition_system(opts, n_worker):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, 'trained_system.npz'))
    dictionary = trained_system['dictionary']

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']

    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)
    # ----- TODO -----
    conf = np.zeros((8, 8), dtype=int)

    for i in range(len(test_files)):
        #Open Image
        img_path = join(opts.data_dir, test_files[i])
        img = Image.open(img_path)
        img = np.array(img).astype(np.float32)/255

        #Get words, features, and distance
        wordmap = visual_words.get_visual_words(opts, img, dictionary)
        test_feat = get_feature_from_wordmap_SPM(opts, wordmap)
        hist_dist = distance_to_set(test_feat, trained_system['features'])
        
        #Calculate confusion matrix
        predicted_label = trained_system['labels'][np.argmin(hist_dist)]
        actual_label = test_labels[i]
        conf[actual_label, predicted_label] += 1

    #Calculate accuracy
    acc = np.trace(conf) / np.sum(conf)

    return conf, acc

