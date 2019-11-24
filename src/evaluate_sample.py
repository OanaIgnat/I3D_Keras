'''
Loads pretrained model of I3d Inception architecture for the paper: 'https://arxiv.org/abs/1705.07750'
Evaluates a RGB and Flow sample similar to the paper's github repo: 'https://github.com/deepmind/kinetics-i3d'
'''
import os
from os import path

import numpy as np
from i3d_inception import Inception_Inflated3d
import tensorflow as tf
from tqdm import tqdm
import time

# INPUT_SHAPE = 55
FRAME_HEIGHT = 224
FRAME_WIDTH = 224
NUM_RGB_CHANNELS = 3
NUM_FLOW_CHANNELS = 2

NUM_CLASSES = 400

LABEL_MAP_PATH = 'data/label_map.txt'


def load_model(include_top_value):
    # build model for RGB data
    # and load pretrained weights (trained on kinetics dataset only)
    rgb_model = Inception_Inflated3d(
        include_top=include_top_value,
        weights='rgb_kinetics_only',
        input_shape=(INPUT_SHAPE, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS),
        classes=NUM_CLASSES)
    return rgb_model


def save_data(video_name, rgb_model, rgb_sample):
    # make prediction
    start_time = time.time()
    rgb_logits = rgb_model.predict(rgb_sample)
    print("---rgb_model.predict(rgb_sample):  %s seconds ---" % (time.time() - start_time))

    # path_output_features = "data/results_features/"
    path_output_features = "data/results_logits/"
    if not os.path.exists(path_output_features):
        os.makedirs(path_output_features)
    start_time = time.time()
    np.save(path_output_features + video_name + ".npy", rgb_logits)
    print("---np.save:  %s seconds ---" % (time.time() - start_time))

    return


if __name__ == '__main__':

    set_video_names = set()
    video_path = "data/results/"
    # path_output_features = "data/results_features_3s/"
    path_output_features = "data/results_logits/"

    for filename in os.listdir(video_path):
        video_name = "_".join(filename.split("_")[:-1])
        if filename.endswith((".npy")) and not path.exists(path_output_features + video_name + ".npy"):
            set_video_names.add(video_name)

    rgb_sample = np.load("data/results/" + list(set_video_names)[0] + "_rgb.npy")
    INPUT_SHAPE = rgb_sample.shape[1]
    print(INPUT_SHAPE)

    start_time = time.time()
    # rgb_model = load_model(include_top_value=False)
    # to get logits
    rgb_model = load_model(include_top_value=True)
    print("---load_model:  %s seconds ---" % (time.time() - start_time))

    for video_name in tqdm(list(set_video_names)):
        print("----- Processing " + video_name)
        # load RGB sample (just one example)
        rgb_sample = np.load("data/results/" + video_name + "_rgb.npy")
        INPUT_SHAPE = rgb_sample.shape[1]
        if INPUT_SHAPE < 8:
            print("data/results/" + video_name + "_rgb.npy" + "smaller than 8s")
            continue

        save_data(video_name, rgb_model, rgb_sample)
