'''
Trains a simple convnet to recognise a smile
'''

from __future__ import print_function

from os.path import exists

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Activation, SeparableConv2D, GlobalAveragePooling2D
from keras import backend as K
from keras.applications import *
from keras.models import load_model
import keras.backend as K

import numpy as np
import pandas as pd
from PIL import Image

import argparse

from data_utils import load_data_to_labels
from data_utils import generate_data
from data_utils import Plotter

batch_size = 64
num_target_values = 2
epochs = 20
steps_per_epoch = 32



def mean_absolute_error(y_true, y_pred):
    return K.mean(np.absolute(y_true - y_pred))



def create_model(input_shape, all_layers_trainable = False):
    conv_base = Xception(input_shape = input_shape, include_top = False, weights = 'imagenet')
    is_layer_trainable = False
    for layer in conv_base.layers:
        if layer.name == 'block14_sepconv1': # we can start with some other
            is_layer_trainable = True
        else:
            is_layer_trainable = all_layers_trainable

        layer.trainable = is_layer_trainable

    model = Sequential()
    model.add(conv_base)
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(num_target_values, name="prediction"))

    return (model)


def make_all_layers_trainable(mdl, is_trainable = True):
    for layer in mdl.layers:
        layer.trainable = is_trainable

    return (mdl)




def train(patch_size, label_path, image_path, train_all_layers):
    # input image dimensions: TODO get from data or command-line params
    input_shape = (patch_size, patch_size, 3)

    train, test, valid = load_data_to_labels(label_path, train_fraction = 0.7, test_fraction = 0.15)

    train_len = len(train)
    test_len = len(test)
    valid_len = len(valid)
    print('Input data: train_len: ' + str(train_len) + ", test_len: " + str(test_len) + ", valid_len: " + str(valid_len))



    model = None
    if exists("best_student.mdl"):
        print("Found a pre-trained model, so loading that")
        model = load_model("best_student.mdl")
        print("Making all layers trainable")
        model = make_all_layers_trainable(model)
    else:
        print("No pre-trained model found, creating a new one")
        model = create_model(input_shape)

    if train_all_layers:
        model = make_all_layers_trainable(model)

    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=keras.optimizers.RMSprop(lr=0.01, clipnorm=1),
    #              optimizer=keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['mae'])

    #if exists("student.mdl"):
    #    model.load_weights("student.mdl")

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_mean_absolute_error')
    early_stop = keras.callbacks.EarlyStopping(monitor='val_mean_absolute_error', patience=10)
    model_checkpoint = keras.callbacks.ModelCheckpoint("best_student.mdl", save_best_only=True, monitor='val_mean_absolute_error')
    no_nan = keras.callbacks.TerminateOnNaN()
    tb = keras.callbacks.TensorBoard()
    plotter = Plotter(input_shape[0])

    # TODO New callback to do sample inference after each epoch
    model.fit_generator(generate_data(image_path, train, batch_size, patch_size),
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                callbacks=[reduce_lr,early_stop,model_checkpoint, no_nan, tb, plotter],
                verbose=1,
                validation_data=generate_data(image_path, valid, batch_size, patch_size),
                validation_steps=valid_len/batch_size)

    model.save("transfer_student.mdl")

    print('Evaluation on latest model:')
    score = model.evaluate_generator(generate_data(image_path, test, batch_size, patch_size), steps=16)
    print('\tTest loss:', score[0])
    print('\tTest MSE:', score[1])

    best_model = keras.models.load_model('best_student.mdl')
    print('Evaluation on best model:')
    score = best_model.evaluate_generator(generate_data(image_path, test, batch_size, patch_size), steps=16)
    print('\tTest loss:', score[0])
    print('\tTest MSE:', score[1])


    num_test_samples = 512
    # evaluating on latest model
    print("\nCalculating error over " + str(num_test_samples) + " test samples... (using latest model)")
    predictions = model.predict_generator(generate_data(image_path, test, num_test_samples, patch_size), steps=1)
    np.save("predictions_latest.npy", predictions)

    # Make it easy to compare predictions to actuals for test data
    test_vs_predict = []
    for i in range(0, num_test_samples):
        sample = {"file":test[i][0],
                  "brightness":test[i][1], "sharpness":test[i][2],
                  "brightness_student":predictions[i][0], "sharpness_student":predictions[i][1]}
        test_vs_predict.append(sample)

    print("Saving errors on " + str(num_test_samples) + " to csv")
    df = pd.DataFrame(test_vs_predict)
    df.to_csv("predictions_vs_test_latest.csv")

    # evaluating on best model
    print("\nCalculating error over " + str(num_test_samples) + " test samples... (using best model)")
    predictions = best_model.predict_generator(generate_data(image_path, test, num_test_samples, patch_size), steps=1)
    np.save("predictions_best.npy", predictions)

    # Make it easy to compare predictions to actuals for test data
    test_vs_predict = []
    for i in range(0, num_test_samples):
        sample = {"file":test[i][0],
                  "brightness":test[i][1], "sharpness":test[i][2],
                  "brightness_student":predictions[i][0], "sharpness_student":predictions[i][1]}
        test_vs_predict.append(sample)

    print("Saving errors on " + str(num_test_samples) + " to csv")
    df = pd.DataFrame(test_vs_predict)
    df.to_csv("predictions_vs_test_best.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", help="Path to images to train on", type=str, required=True)
    parser.add_argument("--patchsize", help="Dimensions for input image", type=int, required=False, default=299)
    parser.add_argument("--labels", help="File containing training labels", type=str, default="image_to_smile.json")
    parser.add_argument("--train")
    parser.add_argument("--trainall", action='store_true',help="Train all layers")
    args = parser.parse_args()
    
    train(args.patchsize, args.labels, args.images, args.trainall)
