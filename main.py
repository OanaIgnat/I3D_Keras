from preprocess import save_as_npy_file
import numpy as np
import os

if __name__ == "__main__":
    root = "/local/oignat/Action_Recog/vlog_action_recognition/data/Video/YOLO/miniclips_results/"

    video_name = "1p0_1mini_2"
    origin_path = root + video_name + "/frames/"
    rgb_path = 'data/' + video_name + '_rgb.npy'
    flow_path = 'data/' + video_name + '_flow.npy'
    if not os.path.isfile(origin_path):
        result = save_as_npy_file(origin_path)
        np.save(rgb_path, result)

    SAMPLE_DATA_PATH = {
        'rgb' : rgb_path,
        'flow' : flow_path
    }
    os.system('python evaluate_sample.py --eval-type rgb --video ' + video_name)
