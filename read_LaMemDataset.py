__author__ = 'charlie'
import numpy as np
import os
import random
from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile
import glob

import TensorflowUtils as utils

DATA_URL = 'http://memorability.csail.mit.edu/lamem.tar.gz'


def read_dataset(data_dir):
    pickle_filename = "lamem.pickle"
    pickle_filepath = os.path.join(data_dir, pickle_filename)
    if not os.path.exists(pickle_filepath):
        utils.maybe_download_and_extract(data_dir, DATA_URL, is_tarfile=True)
        lamem_folder = (DATA_URL.split("/")[-1]).split(os.path.extsep)[0]
        result = {'images': create_image_lists(os.path.join(data_dir, lamem_folder))}
        print ("Pickling ...")
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
    else:
        print ("Found pickle file!")

    with open(pickle_filepath, 'rb') as f:
        result = pickle.load(f)
        training_records = result['images']
        del result

    return training_records


def create_image_lists(image_dir):
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    image_list = []

    file_list = []
    file_glob = os.path.join(image_dir, "images", '*.' + 'jpg')
    file_list.extend(glob.glob(file_glob))

    if not file_list:
        print('No files found')
    else:
        image_list = file_list

    random.shuffle(image_list)
    no_of_images = len(image_list)
    print ('No. of Image files: %d' % no_of_images)

    return image_list
