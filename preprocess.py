import cv2
import os
import numpy as np

SMALLEST_DIM = 256
IMAGE_CROP_SIZE = 224
FRAME_RATE = 25


# sample frames at 25 frames per second
def sample_video(video_path, path_output):
    for filename in os.listdir(video_path):
        if filename.endswith((".mp4", ".avi")):
            filename = video_path + filename
            os.system("ffmpeg -r {1} -i {0} -q:v 2 {2}/frame_%05d.jpg".format(filename, FRAME_RATE,
                                                                                                  path_output))
        else:
            continue


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
    for full_file_path in sorted_list_frames:
        img = cv2.imread(full_file_path, cv2.IMREAD_UNCHANGED)
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


def read_frames(video_path):
    list_frames = []
    for file in os.listdir(video_path):
        if file.endswith(".jpg"):
            full_file_path = video_path + file
            list_frames.append(full_file_path)
    sorted_list_frames = sorted(list_frames)
    return sorted_list_frames


def run_flow(sorted_list_frames):
    sorted_list_img = []
    for frame in sorted_list_frames:
        img = cv2.imread(frame, cv2.IMREAD_UNCHANGED)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sorted_list_img.append(img_gray)

    result = np.zeros((1, IMAGE_CROP_SIZE, IMAGE_CROP_SIZE, 2))
    prev = sorted_list_img[0]
    for curr in sorted_list_img[1:]:
        flow = compute_optical_flow(prev, curr)
        flow = pre_process_flow(flow)
        prev = curr
        result = np.append(result, flow, axis=0)

    result = result[1:, :, :, :]
    result = np.expand_dims(result, axis=0)
    return result


#  For the Flow data, we added a third channel of all 0, then added 0.5 to the entire array, so that results are also between 0 and 1
def show_flow(video_path_npy):

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
    video = cv2.VideoWriter("data/" + video_name + ".avi", fourcc, fps, (width, heigth))

    for i in range(0, nb_frames):
        flow_extra_1st_img = flow_extra[i, :, :, :]
        # confirm the normalization
        # print('flow Min: %.3f, Max: %.3f' % (flow_extra_1st_img.min(), flow_extra_1st_img.max()))

        flow_extra_1st_img *= 255.0 / flow_extra_1st_img.max()
        flow_extra_1st_img = np.uint8(flow_extra_1st_img)
        video.write(flow_extra_1st_img)

        # cv2.imshow('img_flow', flow_extra_1st_img)
        # cv2.waitKey(0)
    cv2.destroyAllWindows()
    video.release()


# From the RGB data, we added 1 and then divided by 2 to rescale between 0 and 1
def show_rgb(video_path_npy):
    rgb = np.load(video_path_npy)
    rgb = (rgb + 1) / 2
    rgb = np.squeeze(rgb, axis=0)  # remove the batch dimension
    nb_frames = rgb.shape[0]
    width = rgb.shape[1]
    heigth = rgb.shape[2]

    video_name = video_path_npy.split("/")[-1][:-4]
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    fps = 10
    video = cv2.VideoWriter("data/" + video_name + ".avi", fourcc, fps, (width, heigth))

    for i in range(0, nb_frames):
        rgb_1st_img = rgb[i, :, :, :]
        # confirm the normalization
        # print('rgb Min: %.3f, Max: %.3f' % (rgb.min(), rgb.max()))
        rgb_1st_img *= 255.0 / rgb_1st_img.max()
        rgb_1st_img = np.uint8(rgb_1st_img)
        video.write(rgb_1st_img)

    cv2.destroyAllWindows()
    video.release()

def pre_process_flow(flow_frame):
    resized = resize(flow_frame)
    img_cropped = crop_center(resized, (IMAGE_CROP_SIZE, IMAGE_CROP_SIZE))
    new_img = np.reshape(img_cropped, (1, IMAGE_CROP_SIZE, IMAGE_CROP_SIZE, 2))
    return new_img


#  Pixel values are truncated to the range [-20, 20], then rescaled between -1 and 1
def compute_optical_flow(prev, curr):
    optical_flow = cv2.optflow.createOptFlow_DualTVL1()
    flow_frame = optical_flow.calc(prev, curr, None)
    flow_frame = np.clip(flow_frame, -20, 20)
    flow_frame = flow_frame / 20.0
    return flow_frame


if __name__ == "__main__":
    path_output = "data/frames/"
    video_path = "data/"
    if not os.path.exists(path_output):
        os.makedirs(path_output)
        sample_video(video_path, path_output)
    # path_output = "/local/oignat/Action_Recog/vlog_action_recognition/data/Video/YOLO/miniclips_results/1p0_1mini_1/frames/"

    sorted_list_frames = read_frames(path_output)
    # result = run_rgb(sorted_list_frames)
    # np.save('data/cricket_rgb.npy', result)
    # show_rgb('data/cricket_rgb.npy')

    flow = run_flow(sorted_list_frames)
    np.save('data/cricket_flow.npy', flow)
    show_flow('data/cricket_flow.npy')
