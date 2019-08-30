import argparse

import cv2
import os
import numpy as np
from preprocess import IMAGE_CROP_SIZE


#  For the Flow data, we added a third channel of all 0, then added 0.5 to the entire array, so that results are also between 0 and 1
def show_flow(video_path_npy):
    flow = np.load(video_path_npy)
    flow = np.squeeze(flow, axis=0)  # remove the batch dimension
    nb_frames = flow.shape[0]

    flow_extra = np.zeros((nb_frames, IMAGE_CROP_SIZE, IMAGE_CROP_SIZE, 3))
    flow_extra[:, :, :, :-1] = flow
    flow_extra = (flow_extra + 1) / 2

    for i in range(0, nb_frames):
        flow_extra_1st_img = flow_extra[i, :, :, :]
        cv2.imshow('img_flow', flow_extra_1st_img)
        cv2.waitKey(0)


# From the RGB data, we added 1 and then divided by 2 to rescale between 0 and 1
def show_rgb(video_path_npy):
    rgb = np.load(video_path_npy)
    rgb = (rgb + 1) / 2
    rgb = np.squeeze(rgb, axis=0)  # remove the batch dimension
    nb_frames = rgb.shape[0]

    for i in range(0, nb_frames):
        rgb_1st_img = rgb[i, :, :, :]
        cv2.imshow('img_flow', rgb_1st_img)
        cv2.waitKey(0)


def save_rgb_video(video_path_npy, path_output):
    rgb = np.load(video_path_npy)
    rgb = (rgb + 1) / 2
    rgb = np.squeeze(rgb, axis=0)  # remove the batch dimension
    nb_frames = rgb.shape[0]
    width = rgb.shape[1]
    heigth = rgb.shape[2]

    video_name = video_path_npy.split("/")[-1][:-4]
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    fps = 10
    video = cv2.VideoWriter(path_output + video_name + ".avi", fourcc, fps, (width, heigth))

    for i in range(0, nb_frames):
        rgb_1st_img = rgb[i, :, :, :]

        # Need uint8 values for write array in video, so we rescale again the pixel values
        rgb_1st_img *= 255.0 / rgb_1st_img.max()
        rgb_1st_img = np.uint8(rgb_1st_img)
        video.write(rgb_1st_img)

    cv2.destroyAllWindows()
    video.release()


def save_flow_video(video_path_npy, path_output):
    flow = np.load(video_path_npy)
    flow = np.squeeze(flow, axis=0)  # remove the batch dimension
    nb_frames = flow.shape[0]

    flow_extra = np.zeros((nb_frames, IMAGE_CROP_SIZE, IMAGE_CROP_SIZE, 3))
    flow_extra[:, :, :, :-1] = flow
    flow_extra = (flow_extra + 1) / 2

    width = flow.shape[1]
    heigth = flow.shape[2]
    video_name = video_path_npy.split("/")[-1][:-4]
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    fps = 10
    video = cv2.VideoWriter(path_output + video_name + ".avi", fourcc, fps, (width, heigth))

    for i in range(0, nb_frames):
        flow_extra_1st_img = flow_extra[i, :, :, :]

        # Need uint8 values for write array in video, so we rescale again the pixel values
        flow_extra_1st_img *= 255.0 / flow_extra_1st_img.max()
        flow_extra_1st_img = np.uint8(flow_extra_1st_img)
        video.write(flow_extra_1st_img)

    cv2.destroyAllWindows()
    video.release()

def main(args):
    # files created in preprocess.py
    npy_rgb_output = 'data/' + args.video_name + '_rgb.npy'
    npy_flow_output = 'data/' + args.video_name + '_flow.npy'

    if not os.path.exists(args.path_output_video):
        os.makedirs(args.path_output_video)

    show_rgb(npy_rgb_output)
    save_rgb_video(npy_rgb_output, args.path_output_video)

    show_flow(npy_flow_output)
    save_flow_video(npy_flow_output, args.path_output_video)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_name', type=str, default="cricket")
    parser.add_argument('--path_output_video', type=str, default='data/viz_results/')
    args = parser.parse_args()

    main(args)


