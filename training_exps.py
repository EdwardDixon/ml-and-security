#
#  Trains a series of models so you can do a lot of experimenting with a single command.
#  Unified train-from-scratch and transfer learning approaches.
#

from __future__ import print_function


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Activation, SeparableConv2D, GlobalAveragePooling2D
from keras import backend as K
from keras.applications import *
import tensorflow as tf

from datetime import datetime
import json
from os.path import exists
import os

import numpy as np
import pandas as pd

from data_utils import load_data_to_labels, generate_data

import argparse


def create_small_cnn(num_target_values = 2, input_shape = (128,128,3)):
    # K.clear_session()
    # tf.reset_default_graph()

    # Create a little CNN we'll train from scratch, just like smile_student.py
    model = Sequential()
    model.add(SeparableConv2D(32, 3, activation='relu', input_shape=input_shape))
    model.add(SeparableConv2D(64, 3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2))
    model.add(SeparableConv2D(64, 3, activation='relu'))
    model.add(SeparableConv2D(128, 3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2))
    model.add(SeparableConv2D(64, 3, activation='relu'))
    model.add(SeparableConv2D(128, 3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2))
    model.add(SeparableConv2D(64, 3, activation='relu'))
    model.add(SeparableConv2D(128, 3, activation='relu'))
    model.add(MaxPooling2D(2))
    model.add(SeparableConv2D(64, 3, activation='relu'))
    model.add(GlobalAveragePooling2D())

    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(num_target_values, name="prediction"))

    return model

def create_transfer_cnn(num_target_values = 2, input_shape = (128,128,3)):
    # K.clear_session()
    # tf.reset_default_graph()

    # Transfer learning using a big pre-trained model, like transfer_student.py
    conv_base = VGG16(input_shape=input_shape, include_top=False, weights='imagenet')
    is_layer_trainable = False
    for layer in conv_base.layers:
        if layer.name == 'block14_sepconv1':  # we can start with some other
            is_layer_trainable = True
        layer.trainable = is_layer_trainable

    model = Sequential()
    model.add(conv_base)
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(num_target_values, name="prediction"))

    return model


def train_model(args, model, batchsize, precision, train, test, valid,
                patch_size = 128, model_fname = None):
    image_path = args.images
    epochs = args.epochs
    validbatches = args.validbatches
    steps = args.steps
    early_stop = args.earlystop
    auto_steps = args.autosteps

    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=keras.optimizers.RMSprop(lr=0.01, clipnorm=1),
                  #              optimizer=keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['mae'])

    # if exists("student.mdl"):
    #    model.load_weights("student.mdl")

    no_nan = keras.callbacks.TerminateOnNaN()
    tboard = keras.callbacks.TensorBoard()
    callbacks = [no_nan, tboard]
    if (early_stop):
        callbacks.append(keras.callbacks.EarlyStopping(monitor='val_mean_absolute_error', patience=10))

    if (model_fname is not None):
        callbacks.append(keras.callbacks.ModelCheckpoint(model_fname, save_best_only=True,
                                                         monitor='val_mean_absolute_error'))

    valid_steps = validbatches
    train_steps = steps
    test_steps  = 16
    if (auto_steps):
        valid_steps = int(len(valid)/batchsize)
        train_steps = int(len(train)/batchsize)
        test_steps  = int(len(test)/batchsize)

    print('len(train)=' + str(len(train)) + ', len(valid)=' + str(len(valid)) +
          ', len(test)=' + str(len(test)))
    print('train_steps=' + str(train_steps) + ', valid_steps=' + str(valid_steps) +
          ', test_steps=' + str(test_steps))

    model.fit_generator(generate_data(image_path, train, batchsize, patch_size=patch_size,
                                      precision=precision, use_eyes = args.useeyes),
                        steps_per_epoch=train_steps,
                        epochs=epochs,
                        callbacks=callbacks,
                        verbose=1,
                        validation_data=generate_data(image_path, valid, batchsize,
                                                      patch_size=patch_size, use_eyes = args.useeyes),
                        validation_steps=valid_steps)

    # Evaluate &r
    print('Evaluation on latest model:')
    score = model.evaluate_generator(generate_data(image_path, test, batchsize,
                                                   patch_size=patch_size, use_eyes = args.useeyes),
                                     steps=test_steps)

    last_mae = score[1]
    best_mae = -1
    if (model_fname is not None):
        best_model = keras.models.load_model(model_fname)
        score = best_model.evaluate_generator(generate_data(image_path, test, batchsize,
                                                            patch_size=patch_size, use_eyes = args.useeyes),
                                              steps=test_steps)
        best_mae = score[1]

    print('Results: last MAE: ' + str(last_mae) + ", best MAE: " + str(best_mae))

    # MAE
    return [last_mae, best_mae]


def train_models(args):
    # Trains lots of models so we can see effect of reducing precision
    time_now = datetime.now()

    if not os.path.exists("experiments"):
        os.makedirs("experiments")

    cur_time = time_now.strftime("%Y_%m_%d_%H_%M")
    report_name = "experiments/precision_results_" + cur_time + ".csv"

    # Keep track of how the experiment was run
    report_settings = report_name.replace(".csv", "_settings.json")
    fsettings = open(report_settings, "w")
    json.dump(vars(args), fsettings)
    fsettings.close()

    # Write results as we go, in case we need to terminate early
    fout = open(report_name, "w")
    fout.write("index,precision,trial,mae_test_last,mae_test_best\n")

    models_trained = 0

    train, test, valid = load_data_to_labels(args.labels, train_fraction=0.7,
                                             test_fraction=0.15, use_eyes = args.useeyes)

    num_target_values = 2 # for Nose information
    if (args.useeyes):
        num_target_values = 6

    for p in args.precisions:
        for trial in range(1, args.trials + 1):

            # Create model, either using pre-trained or train from scratch
            if args.fromscratch == False:
                model = create_transfer_cnn(num_target_values)
            else:
                model = create_small_cnn(num_target_values)

            batch_size = args.batchsize
            if (args.gpus > 1):
                model = keras.utils.multi_gpu_model(model, gpus=args.gpus)
                batch_size = batch_size * args.gpus

            filename = None
            if (args.savemodels):
                filename = "experiments/model_" + cur_time + "_" + str(p) + ".model"

            # Train model
            mae_test = train_model(args, model, batch_size, p, train, test, valid,
                                   model_fname=filename)

            # Write results as we go, in case we need to stop early
            fout.write(str(models_trained) + "," + str(p) + "," + str(trial) + "," +
                       str(mae_test[0]) + "," + str(mae_test[1]) + "\n")
            fout.flush()

            models_trained = models_trained + 1


    fout.close()

    return (models_trained, report_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=str, help="Path to the image files")
    parser.add_argument("--labels", default="image_to_smile.json", type=str, help = "Path to labels JSON")
    parser.add_argument("--batchsize", default=64, type=int, help="How many samples per batch?")
    parser.add_argument("--steps", default=1000, type=int, help="How many steps per epoch?")
    parser.add_argument("--epochs", default=100, type=int, help="How many epochs to train for?")
    parser.add_argument("--gpus", default=1, type=int, help="How many GPUs to use?")
    parser.add_argument("--validbatches", default=8, type=int, help="How many batches to use for validation?  More=more accurate error estimate, but also more computational cost.")
    parser.add_argument("--precisions", nargs='+', type=int, default=[8], required=False, help="List of decimal places to try (mitigation/ablation), e.g. -1 0 1 2 3 4")
    parser.add_argument("--trials", type=int, default=1, required=False, help="How many models to train at each level of precision?  Quality varies, due to random initialization of weights.")
    parser.add_argument("--fromscratch", action='store_true', help="Use transfer learning or train from scratch?")
    parser.add_argument("--earlystop", action='store_true', help="Use early stop callback")
    parser.add_argument("--savemodels", action='store_true', help="Save best models for every precision")
    parser.add_argument("--autosteps", action='store_true', help="Automatically calculate the number of steps based on available data")
    parser.add_argument("--useeyes", action='store_true', help="Use information about eyes")

    args = parser.parse_args()

    num_models_trained, report_name = train_models(args)
    print("Trained " + str(num_models_trained) + " model(s), detailed report in " + report_name)
    exit(0)
    return


if __name__ == "__main__":
    main()
    exit(-1)
