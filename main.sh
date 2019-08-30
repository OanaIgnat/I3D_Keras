#!/bin/sh

#pip install -r requirements.txt

python preprocess.py
python evaluate_sample.py
python visualize.py