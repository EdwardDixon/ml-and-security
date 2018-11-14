# Given a face image and a model, creates a new image plotting the nose coordinates (or what the model thinks is the nose!)
from __future__ import print_function


import keras
from PIL import Image
import numpy as np
from data_utils import *

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Path to image to process", type=str, required=True)
    parser.add_argument("--model", help="Model file to load", type=str, required=True)
    parser.add_argument("--output", help="Path to output file (the input image, resized, with a green cross on the nose)", type=str, default="nose_prediction.png")
    parser.add_argument("--size", help="Image size in pixels (default = 182)", type=int, default=182)
    args = parser.parse_args()

    input_shape = (args.size, args.size, 3)

    img = Image.open(args.image)
    img = img.resize((input_shape[0], input_shape[1]), Image.ANTIALIAS)

    data = []
    data.append(np.asarray(img))
    data = np.asarray(data)

    data = data / 255.0
    data = data - 0.5

    result = None
    mdl = keras.models.load_model(args.model)
    result = mdl.predict(data)
    print("Model output: " + str(result[0]))
    #nose_coords = ((int)(input_shape[0] * result[0][0]), int(input_shape[1] * result[0][1]))
    write_image_result(img, args.size, result[0], args.output)

if __name__ == "__main__":
    main()
