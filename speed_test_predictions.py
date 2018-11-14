# Written to help answer the question "How cheaply can we support 1,000 queries?"
from __future__ import print_function


import keras
from PIL import Image
import numpy as np
import timeit

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Path to image to process", type=str, required=True)
    parser.add_argument("--model", help="Model file to load", type=str, required=True)
    parser.add_argument("--numtrials", help="How many predictons to make (same image each time, just a speed test)", type=int, default=256)

    args = parser.parse_args()

    input_shape = (128, 128, 3)

    img = Image.open(args.image)
    img = img.resize((input_shape[0], input_shape[1]), Image.ANTIALIAS)

    data = []
    data.append(np.asarray(img))
    data = np.asarray(data)

    data = data / 255.0
    data = data - 0.5

    start = timeit.default_timer()

    result = None
    for i in range(0, args.numtrials):
        mdl = keras.models.load_model(args.model)
        result = mdl.predict(data)
        print(result)

    stop = timeit.default_timer()
    elapsed = stop - start

    time_for_thousands = (1000 / args.numtrials) * elapsed

    print("Processed "+ str(args.numtrials) + " images in " + str(stop - start) + " seconds, would take " + str(time_for_thousands) + " seconds to do 1,000 - " + str(time_for_thousands/3600) + " hours")

    nose_coords = ((int)(input_shape[0] * result[0][0]), int(input_shape[1] * result[0][1]))
    for i in range(-5,5):
        img.putpixel((nose_coords[0] + i, nose_coords[1]), (0,255,0))
        img.putpixel((nose_coords[0], nose_coords[1] + i), (0, 255, 0))

    img.save(open("prediction.jpg", "wb"))

if __name__ == "__main__":
    main()
