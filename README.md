# Kinetics-I3D in Keras

Keras implementation of I3D video action detection method reported in the paper [Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/abs/1705.07750).

Original implementation by the authors can be found in this [repository](https://github.com/deepmind/kinetics-i3d), together with details about the pre-processing techniques.

The __model architecture__ is based on this [repository](https://github.com/dlpbc/keras-kinetics-i3d).

The __optical flow__ computation and all the __preprocesing__ is done by myself, following the indications from the original paper: see details about the [preprocessing techniques](https://github.com/deepmind/kinetics-i3d#sample-data-and-preprocessing).

# Usage
```
sh main.sh
```

This script runs all the modules: `video preprocessing`, `model architecture` and `visualization of results`
and installs all the required libraries.
```
python preprocess.py

python evaluate_sample.py

python visualize.py
```


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

# License
- All code in this repository are licensed under the MIT license as specified by the LICENSE file.
- The i3d (rgb and flow) pretrained weights were ported from the ones released [Deepmind](https://deepmind.com) in this [repository](https://github.com/deepmind/kinetics-i3d) under [Apache-2.0 License](https://github.com/deepmind/kinetics-i3d/blob/master/LICENSE)
