# Kinetics-I3D in Keras

Keras implementation of I3D video action detection method reported in the paper [Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/abs/1705.07750).

Original implementation by the authors can be found in this [repository](https://github.com/deepmind/kinetics-i3d), together with details about the pre-processing techniques.

The __model architecture__ is based on this [repository](https://github.com/dlpbc/keras-kinetics-i3d).

The __optical flow__ computation and all the __preprocesing__ is done by myself, following the indications from the original paper: see details about the [preprocessing techniques](https://github.com/deepmind/kinetics-i3d#sample-data-and-preprocessing).

# Data

Under [`data/input_videos`](data/input_videos) there is an input video.

After running the preprocessing module you will obtain all the video frames in [`data/frames`](data/frames) (sampled at 25 fps, as specified in the original paper).

The results (the preprocessed rgb video and the optical flow video) are saved under [`data/viz_results`](data/viz_results):


![Alt Text](data/gifs/cricket_rgb.gif)

![Alt Text](data/gifs/cricket_flow.gif)

# Usage
```
bash main.sh
```

This script runs all the modules: `video preprocessing`, `model architecture` and `visualization of results`
and installs all the required libraries.


### More details

With default flags settings, the `evaluate_sample.py` script builds two I3d Inception architecture (2 stream: RGB and Optical Flow), loads their respective pretrained weights and evaluates RGB sample and Optical Flow sample obtained from video data.

You can set flags to evaluate model using only one I3d Inception architecture (RGB or Optical Flow) as shown below:

```
# For RGB
python evaluate_sample.py --eval-type rgb

# For Optical Flow
python evaluate_sample.py --eval-type flow
```

Addtionally, as described in the paper (and the authors repository), there are __two types of pretrained weights__ for RGB and Optical Flow models respectively. These are;
- RGB I3d Inception:
    - Weights Pretrained on Kinetics dataset only
    - Weights pretrained on Imagenet and Kinetics datasets
- Optical Flow I3d Inception:
    - Weights Pretrained on Kinetics dataset only
    - Weights pretrained on Imagenet and Kinetics datasets

The above usage examples loads weights pretrained on Imagenet and Kinetics datasets. To load weight pretrained on Kinetics dataset only add the flag **--no-imagenet-pretrained** to the above commands. See an example below:

```

# RGB I3d Inception model pretrained on kinetics dataset only
python evaluate_sample.py --eval-type rgb --no-imagenet-pretrained
```

# Results

The script outputs the __norm of the logits tensor__, as well as the __top 20 Kinetics classes predicted__ by the model
with their probability and logit values. Using the default flags, the output should resemble the following up to differences in numerical precision:

```
Norm of logits: 144.034286

Top 20 classes and probabilities
0.9999996 38.700874 playing cricket
3.2525813e-07 23.762228 hurling (sport)
8.824412e-08 22.457716 playing tennis
4.547956e-09 19.492287 playing squash or racquetball
4.257603e-09 19.426315 hitting baseball
2.1000581e-09 18.719574 catching or throwing baseball
7.671558e-10 17.712543 catching or throwing softball
3.830608e-10 17.018047 playing badminton
3.4193437e-10 16.904472 shooting goal (soccer)
3.0672612e-10 16.795809 dodgeball
1.6208218e-10 16.157957 playing kickball
6.809785e-11 15.290799 passing American football (in game)
3.362223e-11 14.585041 celebrating
3.0167153e-11 14.476607 shot put
2.2506778e-11 14.18367 hammer throw
1.966535e-11 14.048712 tai chi
1.7816216e-11 13.949963 sword fighting
1.6389917e-11 13.86652 throwing discus
1.1673846e-11 13.5272045 kicking field goal
1.0559041e-11 13.426836 javelin throw

```


# Requirements

```
pip install -r requirements.txt
```
- Keras
- Keras Backend: Tensorflow (tested) or Theano (not tested) or CNTK (not tested)
- h5py
- numpy
- opencv-python
- opencv-contrib-python

I'm using Python 3.6, not sure if it works with Python 2.7

If something does not work, please let me know. I'm happy to help. :)

# License
- All code in this repository are licensed under the MIT license as specified by the LICENSE file.
- The i3d (rgb and flow) pretrained weights were ported from the ones released [Deepmind](https://deepmind.com) in this [repository](https://github.com/deepmind/kinetics-i3d) under [Apache-2.0 License](https://github.com/deepmind/kinetics-i3d/blob/master/LICENSE)

