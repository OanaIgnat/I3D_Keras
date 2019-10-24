'''
Loads pretrained model of I3d Inception architecture for the paper: 'https://arxiv.org/abs/1705.07750'
Evaluates a RGB and Flow sample similar to the paper's github repo: 'https://github.com/deepmind/kinetics-i3d'
'''
import os
from os import path

import numpy as np
import argparse
from preprocess import IMAGE_CROP_SIZE, ROOT_PATH, remove_options

from i3d_inception import Inception_Inflated3d
import tensorflow as tf
from tqdm import tqdm

# INPUT_SHAPE = 55
FRAME_HEIGHT = 224
FRAME_WIDTH = 224
NUM_RGB_CHANNELS = 3
NUM_FLOW_CHANNELS = 2

NUM_CLASSES = 400

LABEL_MAP_PATH = 'data/label_map.txt'

def main(args):

    SAMPLE_DATA_PATH = {
        # 'rgb' : 'data/v_CricketShot_g04_c01_rgb.npy',
        # 'rgb': 'data/results_video/' + args.video_name + "_rgb.npy",
        # 'flow': 'data/results_video/'+ args.video_name + "_flow.npy"
        'rgb':  "data/results/" + args.video_name + "_rgb.npy",
        'flow':  "data/results/" + args.video_name + "_flow.npy"

    }

    # load the kinetics classes
    kinetics_classes = [x.strip() for x in open(LABEL_MAP_PATH, 'r')]

    if args.eval_type in ['rgb', 'joint']:

        # load RGB sample (just one example)
        rgb_sample = np.load(SAMPLE_DATA_PATH['rgb'])
        INPUT_SHAPE = rgb_sample.shape[1]

        if args.no_imagenet_pretrained:
            # build model for RGB data
            # and load pretrained weights (trained on kinetics dataset only) 
            rgb_model = Inception_Inflated3d(
                include_top=args.include_top,
                weights='rgb_kinetics_only',
                input_shape=(INPUT_SHAPE, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS),
                classes=NUM_CLASSES)
        else:
            # build model for RGB data
            # and load pretrained weights (trained on imagenet and kinetics dataset)
            rgb_model = Inception_Inflated3d(
                include_top=args.include_top,
                weights='rgb_imagenet_and_kinetics',
                input_shape=(INPUT_SHAPE, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS),
                classes=NUM_CLASSES)

        # make prediction
        rgb_logits = rgb_model.predict(rgb_sample)
        if not args.include_top:
            path_output_features = "data/results_features/"
            if not os.path.exists(path_output_features):
                os.makedirs(path_output_features)
            np.save(path_output_features + args.video_name + ".npy", rgb_logits)

            return


    if args.eval_type in ['flow', 'joint']:

        # load flow sample (just one example)
        flow_sample = np.load(SAMPLE_DATA_PATH['flow'])
        INPUT_SHAPE = flow_sample.shape[1]

        if args.no_imagenet_pretrained:
            # build model for optical flow data
            # and load pretrained weights (trained on kinetics dataset only)
            flow_model = Inception_Inflated3d(
                include_top=args.include_top,
                weights='flow_kinetics_only',
                input_shape=(INPUT_SHAPE, FRAME_HEIGHT, FRAME_WIDTH, NUM_FLOW_CHANNELS),
                classes=NUM_CLASSES)
        else:
            # build model for optical flow data
            # and load pretrained weights (trained on imagenet and kinetics dataset)
            flow_model = Inception_Inflated3d(
                include_top=args.include_top,
                weights='flow_imagenet_and_kinetics',
                input_shape=(INPUT_SHAPE, FRAME_HEIGHT, FRAME_WIDTH, NUM_FLOW_CHANNELS),
                classes=NUM_CLASSES)
        
        # make prediction
        flow_logits = flow_model.predict(flow_sample)
        if not args.include_top:
            path_output_features = "data/results_features/"
            if not os.path.exists(path_output_features):
                os.makedirs(path_output_features)
            np.save(path_output_features + args.video_name + ".npy", flow_logits)

            return

    # produce final model logits
    if args.eval_type == 'rgb':
        sample_logits = rgb_logits
    elif args.eval_type == 'flow':
        sample_logits = flow_logits
    else: # joint
        sample_logits = rgb_logits + flow_logits

    # produce softmax output from model logit for class probabilities
    sample_logits = sample_logits[0] # we are dealing with just one example
    sample_predictions = np.exp(sample_logits) / np.sum(np.exp(sample_logits))

    sorted_indices = np.argsort(sample_predictions)[::-1]
    print("\nFor video " + args.video_name + ": ")
    print('\nNorm of logits: %f' % np.linalg.norm(sample_logits))
    print('\nTop 5 classes and probabilities')
    for index in sorted_indices[:5]:
        print(sample_predictions[index], sample_logits[index], kinetics_classes[index])

    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parse arguments
    #TODO: add joint (also flow)
    # parser.add_argument('--eval-type',
    #                     help='specify model type. 1 stream (rgb or flow) or 2 stream (joint = rgb and flow).',
    #                     type=str, choices=['rgb', 'flow', 'joint'], default='joint')

    parser.add_argument('--eval-type',
                        help='specify model type. 1 stream (rgb or flow) or 2 stream (joint = rgb and flow).',
                        type=str, choices=['rgb', 'flow', 'joint'], default='rgb')

    parser.add_argument('--no-imagenet-pretrained',
                        help='If set, load model weights trained only on kinetics dataset. Otherwise, load model weights trained on imagenet and kinetics dataset.',
                        action='store_true')

    parser.add_argument('--include_top', type=str, default=False)
    # set_video_names = set()
    # video_path = ROOT_PATH + "data/results_video/"
    # for filename in os.listdir(video_path):
    #     if filename.endswith((".npy")):
    #         video_name = "_".join(filename.split("_")[:-1])
    #         set_video_names.add(video_name)
    # for video_name in list(set_video_names):
    #     parser.add_argument('--video_name', type=str, default=video_name)
    #
    #     args = parser.parse_args()
    #     main(args)
    #
    #     remove_options(parser, ['--video_name'])
    # -------------------------------------------------------------

    # parser.add_argument('--video_name', type=str, default="2Y8XQ")
    # -------------------------------------------------------------

    set_video_names = set()
    video_path = "data/results/"
    path_output_features = "data/results_features/"

    for filename in os.listdir(video_path):
        video_name = "_".join(filename.split("_")[:-1])
        if filename.endswith((".npy")) and not path.exists(path_output_features + video_name + ".npy"):
            set_video_names.add(video_name)

    for video_name in tqdm(list(set_video_names)):
        # channel = "_".join(video_name.split("_")[0:3])
        # if channel != "1p1_2mini_6":
        #     continue
        parser.add_argument('--video_name', type=str, default=video_name)

        args = parser.parse_args()
        main(args)

        remove_options(parser, ['--video_name'])



    args = parser.parse_args()
    main(args)
