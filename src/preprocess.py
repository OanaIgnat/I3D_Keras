import glob
import subprocess
import time

import cv2
import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import argparse

SMALLEST_DIM = 256
IMAGE_CROP_SIZE = 224
FRAME_RATE = 24

if tf.test.is_gpu_available(cuda_only=True):
    ROOT_PATH = "/local2/oignat/i3d_keras/"
else:
    ROOT_PATH = "/local/oignat/Action_Recog/i3d_keras/"
    # ROOT_PATH = "/local2/oignat/large_data/"


# sample frames at 25 frames per second
def sample_video(video_path, path_output):
    # for filename in os.listdir(video_path):
    if video_path.endswith((".mp4", ".avi")):
        # os.system("ffmpeg -r {1} -i {0} -q:v 2 {2}/frame_%05d.jpg".format(video_path, FRAME_RATE,
        #                                                                   path_output))

        os.system("ffmpeg -i {0} -vf fps={1} {2}/frame_%05d.jpg".format(video_path, FRAME_RATE,
                                                                        path_output))


    else:
        raise ValueError("Video path is not the name of video file (.mp4 or .avi)")


# the videos are resized preserving aspect ratio so that the smallest dimension is 256 pixels, with bilinear interpolation
def resize(img):
    # print('Original Dimensions : ', img.shape)

    original_width = int(img.shape[1])
    original_height = int(img.shape[0])

    aspect_ratio = original_width / original_height

    if original_height < original_width:
        new_height = SMALLEST_DIM
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = SMALLEST_DIM
        new_height = int(original_width / aspect_ratio)

    dim = (new_width, new_height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
    # print('Resized Dimensions : ', resized.shape)

    return resized


def crop_center(img, new_size):
    y, x, c = img.shape
    (cropx, cropy) = new_size
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]


def rescale_pixel_values(img):
    # print('Data Type: %s' % img.dtype)
    # print('Min: %.3f, Max: %.3f' % (img.min(), img.max()))
    img = img.astype('float32')
    # normalize to the range 0:1
    # img /= 255.0
    # normalize to the range -1:1
    img = (img / 255.0) * 2 - 1
    # confirm the normalization
    # print('Min: %.3f, Max: %.3f' % (img.min(), img.max()))
    return img


# The provided .npy file thus has shape (1, num_frames, 224, 224, 3) for RGB, corresponding to a batch size of 1
def run_rgb(sorted_list_frames):
    result = np.zeros((1, IMAGE_CROP_SIZE, IMAGE_CROP_SIZE, 3))
    print("Running preprocessing rgb ...")
    for full_file_path in tqdm(sorted_list_frames):
        img = cv2.imread(full_file_path, cv2.IMREAD_UNCHANGED)
        # if not img:
        #     result = np.append(result, result[-1], axis=0)
        #     continue
        img = pre_process_rgb(img)
        new_img = np.reshape(img, (1, IMAGE_CROP_SIZE, IMAGE_CROP_SIZE, 3))
        result = np.append(result, new_img, axis=0)

    result = result[1:, :, :, :]
    result = np.expand_dims(result, axis=0)
    return result


def pre_process_rgb(img):
    resized = resize(img)
    img_cropped = crop_center(resized, (IMAGE_CROP_SIZE, IMAGE_CROP_SIZE))
    img = rescale_pixel_values(img_cropped)
    return img


def read_sorted_frames(video_path):
    list_frames = []
    for file in os.listdir(video_path):
        if file.endswith(".jpg"):
            full_file_path = video_path + "/" + file
            list_frames.append(full_file_path)
    sorted_list_frames = sorted(list_frames)
    return sorted_list_frames


def read_frames(video_path, path_output):
    sorted_list_frames = read_sorted_frames(path_output)
    index = 1
    while len(sorted_list_frames) < 64:
        os.system("ffmpeg -i {0} -vf fps={1} {2}/frame_%05d.jpg".format(video_path, 30 * index,
                                                                        path_output))
        sorted_list_frames = read_sorted_frames(path_output)
        index += 1

    n = len(sorted_list_frames) - 64
    return sorted_list_frames[int(n / 2): int(n / 2) + 64]


def run_flow(sorted_list_frames):
    sorted_list_img = []
    print("Running preprocessing flow: 1. rgb2gray ...")
    for frame in tqdm(sorted_list_frames):
        img = cv2.imread(frame, cv2.IMREAD_UNCHANGED)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sorted_list_img.append(img_gray)

    result = np.zeros((1, IMAGE_CROP_SIZE, IMAGE_CROP_SIZE, 2))
    prev = sorted_list_img[0]
    prev = resize(prev)

    print("Running preprocessing flow: 2. optical flow comp. ...")
    for curr in tqdm(sorted_list_img[1:]):
        curr = resize(curr)
        flow = compute_optical_flow(prev, curr)
        flow = pre_process_flow(flow)
        prev = curr
        result = np.append(result, flow, axis=0)

    result = result[1:, :, :, :]
    result = np.expand_dims(result, axis=0)
    return result


def pre_process_flow(flow_frame):
    # resized = resize(flow_frame)
    img_cropped = crop_center(flow_frame, (IMAGE_CROP_SIZE, IMAGE_CROP_SIZE))
    new_img = np.reshape(img_cropped, (1, IMAGE_CROP_SIZE, IMAGE_CROP_SIZE, 2))
    return new_img


#  Pixel values are truncated to the range [-20, 20], then rescaled between -1 and 1
def compute_optical_flow(prev, curr):
    optical_flow = cv2.optflow.createOptFlow_DualTVL1()
    flow_frame = optical_flow.calc(prev, curr, None)
    flow_frame = np.clip(flow_frame, -20, 20)
    flow_frame = flow_frame / 20.0
    return flow_frame

def main(args):
    # sample all video from video_path at specified frame rate (FRAME_RATE param)
    if not os.listdir(args.path_output):
        sample_video(args.video_path, args.path_output)
    else:
        print("Directory " + args.path_output + " is not empty")

    # make sure the frames are processed in order
    sorted_list_frames = read_frames(args.video_path, args.path_output)

    video_name = args.video_path.split("/")[-1][:-4]
    if not sorted_list_frames:
        print("File " + video_name + " # frames < 10")
        return
    path_output_results = ROOT_PATH + "data/results_overlapping/"
    npy_rgb_output = path_output_results + video_name + '_rgb.npy'

    if not os.path.exists(path_output_results):
        os.makedirs(path_output_results)

    if not os.path.exists(npy_rgb_output) or os.stat(npy_rgb_output).st_size == 0:
        rgb = run_rgb(sorted_list_frames)
        print(rgb.shape)
        np.save(npy_rgb_output, rgb)
    else:
        print("File " + npy_rgb_output + " exists already and it's not empty")

    # npy_flow_output = path_output_results + video_name + '_flow.npy'

    # TODO: Run flow later
    # if not os.path.exists(npy_flow_output) or os.stat(npy_flow_output).st_size == 0:
    #     flow = run_flow(sorted_list_frames)
    #     np.save(npy_flow_output, flow)
    # else:
    #     print("File " + npy_flow_output + " exists already and it's not empty")


def remove_options(parser, options):
    for option in options:
        for action in parser._actions:
            if vars(action)['option_strings'][0] == option:
                parser._handle_conflict_resolve(None, [(option, action)])
                break


def system_call(command):
    p = subprocess.Popen([command], stdout=subprocess.PIPE, shell=True)
    return p.stdout.read()


def create_3s_clips(path_input_video, path_output_video):
    miniclip = path_input_video.split("/")[-1][:-4]
    # path_output_video = "../large_data/10s_clips/"
    if not os.path.exists(path_output_video):
        os.makedirs(path_output_video)

    # consecutive clips
    # command = "ffmpeg -i " + path_input_video + " -acodec copy -f segment -segment_time 3 -vcodec copy -reset_timestamps 1 -map 0 " \
    #           + path_output_video + miniclip + "_%03d.mp4"
    # command = "ffmpeg -hide_banner  -err_detect ignore_err -i " + path_input_video + " -r 24 -codec:v libx264  -vsync 1  -codec:a aac  -ac 2  -ar 48k  -f segment   -preset fast  -segment_format mpegts  -segment_time 0.5 -force_key_frames \"expr:gte(t, n_forced * 3)\" " + path_output_video + miniclip + "_%03d.mp4"

    # overlapping clips
    command_clip_length = "ffprobe -i " + path_input_video + " -show_format -v quiet | sed -n 's/duration=//p'"
    clip_length = system_call(command_clip_length)
    clip_length = int(float(clip_length.strip()))
    # 1 second
    for start_clip in range(0, clip_length - 3):
        command = "ffmpeg -ss " + str(
            start_clip) + " -i " + path_input_video + " -t 3.000 " + path_output_video + miniclip + "_{0:03}.mp4".format(start_clip+1)
        os.system(command)


def create_clips(path_input_video, path_output_video, channels):
    list_videos = glob.glob(path_input_video + "*.mp4")

    for video_file in list_videos:
        miniclip = video_file.split("/")[-1][:-4]
        channel = miniclip.split("_")[0]
        if channel not in channels:
            continue
        create_3s_clips(video_file, path_output_video)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser.add_argument('--path_output', type=str, default=root_path + "data/frames/1p0_1mini_1/")
    #
    # parser.add_argument('--video_path', type=str,
    #                     default="../large_data/miniclips_lifestyle_vlogs/miniclips_dataset_new/1p0_1mini_1.mp4")
    # ---------------------------------------------------------------------------------------------------------------

    # video_path = "/local2/oignat/large_data/" + "10s_clips/"

    video_path = "/local2/oignat/3s_clips/"
    path_output = ROOT_PATH + "data/frames/"
    miniclips_path = "/local2/oignat/miniclips/"
    # miniclips_path = "/local/oignat/Action_Recog/temporal_annotation/miniclips/"
    channels = ["1p0", "1p1", "2p0", "2p1", "3p0", "3p1"]
    # channels = ["4p0","4p1", "5p0", "5p1"]
    # channels = ["6p0","6p1", "7p0", "7p1"]
    # channels = ["1p0"]
    for channel in channels:
        # if channel not in ["1p0", "4p0", "6p0"]:
        create_clips(miniclips_path, video_path, channel)
        # print("Created 3s clips for channel " + channel)
        for filename in os.listdir(video_path):
            if filename.endswith((".mp4", ".avi")):
                video_name = "_".join(filename.split("_")[:-1])
                if video_name.split("_")[0] != channel:
                    print(video_name + "not in " + channel)
                    continue

                # path_output_results = ROOT_PATH + "data/results/"
                path_output_results = ROOT_PATH + "data/results_overlapping/"
                npy_rgb_output = path_output_results + filename.split("/")[-1][:-4] + '_rgb.npy'
                if os.path.exists(npy_rgb_output) and os.stat(npy_rgb_output).st_size != 0:
                    print(npy_rgb_output + " does exist!")
                    continue
                else:
                    print(npy_rgb_output + " does not exist!")

                if not os.path.exists(path_output + filename[:-4]):
                    os.makedirs(path_output + filename[:-4])

                parser.add_argument('--path_output', type=str, default=path_output + filename[:-4],
                                    help=argparse.SUPPRESS)

                parser.add_argument('--video_path', type=str, default=video_path + filename, help=argparse.SUPPRESS)

                args = parser.parse_args()

                main(args)

                os.system("rm -r " + path_output + filename[:-4])  # remove frames
                os.system("rm " + video_path + filename)  # remove 10s clips

                remove_options(parser, ['--path_output', '--video_path'])

    # ---------------------------------------------------------------------------------------------------------------

    # path_input_10s_folders = "/local/oignat/Action_Recog/large_data/10s_clips/"
    # path_output_10s_folders = "/local/oignat/Action_Recog/large_data/10s_clips_processed/"
    # dirs = os.listdir(path_input_10s_folders)
    # for dir in dirs:
    #     for filename in os.listdir(path_input_10s_folders + dir):
    #         if filename.endswith((".mp4", ".avi")):
    #             input_video = path_input_10s_folders + dir + "/" + filename
    #
    #             output_folder = path_output_10s_folders + dir + "/" + filename[:-4]
    #             if not os.path.exists(output_folder + "/frames"):
    #                 os.makedirs(output_folder + "/frames")
    #
    #             if not os.path.exists(output_folder + "/results_video"):
    #                 os.makedirs(output_folder + "/results_video")
    #
    #             parser.add_argument('--path_output', type=str, default=output_folder, help=argparse.SUPPRESS)
    #
    #             parser.add_argument('--video_path', type=str, default=input_video, help=argparse.SUPPRESS)
    #
    #             args = parser.parse_args()
    #
    #             main(args)
    #
    #             remove_options(parser, ['--path_output', '--video_path'])
