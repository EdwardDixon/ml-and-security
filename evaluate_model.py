from __future__ import print_function


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras import backend as K

import numpy as np
import pandas as pd

from data_utils import load_data_to_labels
from data_utils import generate_data

import argparse

def main():
    # TODO move paths to command-line arguments
    image_path = "/media/sf_data_1/bp_aug_aligned"
    label_path = "image_to_smile.json"
    patch_size = 32
    # input image dimensions: TODO get from data or command-line params
    input_shape = (patch_size, patch_size, 3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Folder containing input files", type=str, required=True)
    parser.add_argument("--labels", help="JSON file with ground truth", type=str, required=True)
    parser.add_argument("--tilesize", help="Width/height for images (input images get snipped into small tiles)", type=int, default=256)
    parser.add_argument("--train_fraction", help="What fraction of tiles to keep for training (dev & test split remainder)", type=float, default=0.7)

    args = parser.parse_args()
