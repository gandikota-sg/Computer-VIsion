from os.path import join

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import util
import visual_words
import visual_recog
from opts import get_opts
import time


def main():
    opts = get_opts()

    ## Q1.1
    #img_path = join(opts.data_dir, 'aquarium/sun_aztvjgubyrgvirup.jpg')
    #img = Image.open(img_path)
    #img = np.array(img).astype(np.float32)/255
    #filter_responses = visual_words.extract_filter_responses(opts, img)
    #util.display_filter_responses(opts, filter_responses)

    ## Q1.2
    n_cpu = util.get_num_CPU()
    visual_words.compute_dictionary(opts, n_worker=n_cpu)
    
    ## Q1.3
    img_list = ['kitchen/sun_abjllhhfuvcygwtk.jpg', 'laundromat/sun_aiyluzcowlbwxmdb.jpg', 'waterfall/sun_bzkwnudrbegyufkp.jpg', 'park/sun_autgevtgmlodpvpv.jpg']
    for i in range(len(img_list)):
        img_path = join(opts.data_dir, img_list[i])
        img = Image.open(img_path)
        img = np.array(img).astype(np.float32)/255
        dictionary = np.load(join(opts.out_dir, 'dictionary.npy'))
        wordmap = visual_words.get_visual_words(opts, img, dictionary)
        util.visualize_wordmap(wordmap)

    ## Q2.1-2.4
    start_time = time.time()
    n_cpu = util.get_num_CPU()
    visual_recog.build_recognition_system(opts, n_worker=n_cpu)

    ## Q2.5
    n_cpu = util.get_num_CPU()
    conf, accuracy = visual_recog.evaluate_recognition_system(opts, n_worker=n_cpu)
    
    print(conf)
    print(accuracy)
    np.savetxt(join(opts.out_dir, 'confmat.csv'), conf, fmt='%d', delimiter=',')
    np.savetxt(join(opts.out_dir, 'accuracy.txt'), [accuracy], fmt='%g')

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time} seconds")


if __name__ == '__main__':
    main()
