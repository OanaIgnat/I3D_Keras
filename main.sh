#!/bin/sh

#pip install -r requirements.txt

python src/preprocess.py
python src/evaluate_sample.py >> data/results_video.txt
#python src/visualize.py