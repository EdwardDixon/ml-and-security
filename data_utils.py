import json
from os.path import basename, join
from PIL import Image
import numpy as np
from numpy.random import random, randint
from decimal import *
from skimage.transform import resize
import os
import keras

try:
   import cPickle as pickle
except:
   import pickle



def load_data_to_labels(path, train_fraction = 0.8, test_fraction = 0.1,
                        keep_only_filename = True, use_eyes = False):
    # Original JSON from our AWS queries needs to be transformed for convenient processing
    # We just want a list of filenames, each with: brightness, sharpness
    # We end up with train & test lists, splitting at random
    results = {}
    with open(path, "r") as f:
        results = json.load(f)

    train = []
    test = []
    valid = []

    valid_fraction = train_fraction + test_fraction
    for k in results:
        filename = k
        if keep_only_filename == True:
            filename = basename(filename)

        #precision = 20
        getcontext().prec = 105
        brightness = Decimal(results[k]["FaceDetails"][0]["Quality"]["Brightness"])
        sharpness = Decimal(results[k]["FaceDetails"][0]["Quality"]["Sharpness"])
        noseX = -1
        noseY = -1
        eyeLeftX = -1
        eyeLeftY = -1
        eyeRightX = -1
        eyeRightY = -1
        for l in results[k]["FaceDetails"][0]['Landmarks']:
            if l['Type'] == 'nose':
                noseX = l['X']
                noseY = l['Y']
            if l['Type'] == 'eyeLeft':
                eyeLeftX = l['X']
                eyeLeftY = l['Y']
            if l['Type'] == 'eyeRight':
                eyeRightX = l['X']
                eyeRightY = l['Y']

        boundingbox = results[k]["FaceDetails"][0]['BoundingBox']

        #name_to_label = (filename, results[k]["FaceDetails"][0]["Quality"]["Brightness"], results[k]["FaceDetails"][0]["Quality"]["Sharpness"])
        name_to_label = (filename, brightness, sharpness, boundingbox, eyeLeftX, eyeLeftY, eyeRightX, eyeRightY, noseX, noseY)

        rnd = random(1)
        if rnd < train_fraction:
            train.append(name_to_label)
        elif rnd < valid_fraction:
            test.append(name_to_label)
        else:
            valid.append(name_to_label)

    return (train, test, valid)


def get_pickle(fname):
    data = None
    if (os.path.exists(fname) and os.path.getsize(fname) > 0):
        with open(fname, 'rb') as fr:
            try:
                data = pickle.load(fr)
            except:
                data = None
    return data

def generate_data(path, file_to_label, batch_size, patch_size = 32, precision = 10, use_eyes = False):
    # Returns a tensor X of the inputs (4D assuming RGB images) and a tensor Y of the labels.
    # We use the yield pattern so you can work with large folders
    # We use just the face patch from each image
    
    # Becomes the input tensor
    X = []

    # We support variable label widths
    Y = []

    # Loop over images, yield batches every batch_size images
    image_index = 0

    images_found = 0
    images_not_found = 0
    indexes_not_found = 0

    typestr = "_nose"
    if (use_eyes):
       typestr = "_eyes"

    while(1):
        try:
            image_path = join(path, file_to_label[image_index][0])
            pkl_path = image_path + "_" + str(patch_size) + typestr + ".pickle"
            #save_data = get_pickle(pkl_path)
            save_data = None
            x_shift = 0.0
            y_shift = 0.0
            patch = None

            if (save_data is None):
               img = Image.open(image_path)
               sample_dims = img.size
               img = np.asarray(img)

               images_found = images_found + 1 #assuming the image should have been found

               # We extract a patch from random location in image
               '''
               x_offset = int(file_to_label[image_index][3]["Left"] * sample_dims[0])
               y_offset = int(file_to_label[image_index][3]["Top"] * sample_dims[1])
               width = int(file_to_label[image_index][3]["Width"] * sample_dims[0])
               height = int(file_to_label[image_index][3]["Height"] * sample_dims[1])
               
               if x_offset + width >= sample_dims[0]:
                  x_offset = sample_dims[0] - width
                  
               if y_offset + height >= sample_dims[1]:
                  y_offset = sample_dims[1] - height

               if x_offset < 0:
                  x_offset = 0

               if y_offset < 0:
                  y_offset = 0

               x_shift = float(x_offset)/sample_dims[0]
               y_shift = float(y_offset)/sample_dims[1]

               #patch = img[x_offset:x_offset + patch_size, y_offset:y_offset + patch_size, :]
               #patch = np.asarray(Image.fromarray(patch, 'RGB').resize((patch_size, patch_size,)))
               '''
               patch = img
            
               # Normalize image data! Not very well!
               patch = patch / 255.0
               patch = patch - 0.5

               save_data = (patch, x_shift, y_shift)
                
               with open(pkl_path, 'wb') as fw:
                  pickle.dump(save_data, fw, pickle.HIGHEST_PROTOCOL)

            else: # we have data from pickle
               (patch, x_shift, y_shift) = save_data

            X.append(patch)
            # TODO: we need to adjust the coordinates of eyes or nose???
            if use_eyes:
               labels = (round(file_to_label[image_index][1], precision),
                         round(file_to_label[image_index][2], precision),
                         round(file_to_label[image_index][4] - x_shift, precision),  # Left eye x
                         round(file_to_label[image_index][5] - y_shift, precision),  # Left eye y
                         round(file_to_label[image_index][6] - x_shift, precision),  # Right eye x
                         round(file_to_label[image_index][7] - y_shift, precision)   # Right eye y
               )
            else:
               labels = (#round(file_to_label[image_index][1], precision),
                         #round(file_to_label[image_index][2], precision),
                         round(file_to_label[image_index][8] - x_shift, precision),  # Nose x
                         round(file_to_label[image_index][9] - y_shift, precision)   # Nose y
               )
            Y.append(labels)


            image_index = image_index + 1

        except IOError:
            images_not_found = images_not_found + 1
            #print("IOError, " + "images found: " + str(images_found) + ", images not found: " + str(images_not_found) + ", image_index: " + str(image_index) + ", image_path: " + str(image_path))
            image_index = image_index + 1
        except IndexError:
            indexes_not_found = indexes_not_found + 1
            print("IndexError, " + "images found: " + str(images_found) + ", indexes not found: "
                  + str(indexes_not_found)  +  ", image_index: " + str(image_index)
                  + ", image_path: " + str(image_path))
            image_index = image_index + 1
        except FileNotFoundError:
            print("Could not find file " + str(image_path))

        if image_index >= len(file_to_label):
            image_index = 0
        if len(X) == batch_size:
           X = np.asarray(X)
           Y = np.asarray(Y)

           # Pass a response back to our caller
           yield(X,Y)

           # Reset for next batch
           X = []
           Y = []

    return

def write_image_result(img, result, outpath, color=(0,255,0)):
    # Give a PIL image, draws a cross on the nose
    nose_coords = ((int)(img.width * result[0]), int(img.height * result[1]))
    print("Plotting nose at " + str(nose_coords) + " based on result " + str(result))
    for i in range(-5,5):
        img.putpixel((nose_coords[0] + i, nose_coords[1]), color)
        img.putpixel((nose_coords[0], nose_coords[1] + i), color)

    img.save(open(outpath, "wb"))
    return


def normalize_images(images_as_np):
    images_as_np = images_as_np / 255.0
    images_as_np = images_as_np - 0.5
    return (images_as_np)


def load_image_to_np_arr(path, desired_size):
    img = Image.open(path)
    img = img.resize((desired_size, desired_size), Image.ANTIALIAS)
    data = []
    data.append(np.asarray(img))
    data = np.asarray(data)
    data = normalize_images(data)
    return(data)


def predict(mdl, images):
    data = []
    data.append(np.asarray(img))
    data = np.asarray(data)

    data = data / 255.0
    data = data - 0.5

    result = None
    mdl = keras.models.load_model(args.model)
    result = mdl.predict(images)



class Plotter(keras.callbacks.Callback):

    def __init__(self, input_image_size, input_name="woman_with_curly_hair.jpg"):
        self.sample = load_image_to_np_arr(input_name)
        self.sample_image = Image.open(input_name)
        self.input_image_size = input_image_size

    def set_model(self, model):
        self.model = model

    def make_sample_prediction(self, output_name):
        # Used to make sample predictions after each epoch
        results = self.model.predict(self.sample_image)
        write_image_result(self.sample_image, self.input_image_size, results[0], output_name)
        return

    def on_epoch_end(self, epoch, logs=None):
        output_name = "nose_at_epoch_" + str(epoch) + ".jpg"
        self.make_sample_prediction(output_name)
        return

