#!/bin/sh

#pip install -r requirements.txt

python src/preprocess.py
python src/evaluate_sample.py
python src/visualize.py