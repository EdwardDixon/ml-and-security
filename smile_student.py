'''
Trains a simple convnet to recognise a smile
'''

from __future__ import print_function

from os.path import exists

import keras
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Activation, SeparableConv2D, GlobalAveragePooling2D
from keras import backend as K
from datetime import datetime

import numpy as np
import pandas as pd

from data_utils import load_data_to_labels
from data_utils import generate_data

print ("** To get us started, we need you to enter a sequence of integers separted by commas**")
print ("** ... the amount of integers entered will indicate how many experiments should be run **")
print ("** ... while the individual numbers indicate the desired precision **")
print ("*****")

#runs_precisions = input("Please enter your dnesired precisions for this set of experiments: ")
#runs_precisions_list = [int(i) for i in runs_precisions.split(',') if i.isdigit()]
runs_precisions_list = [-1]
print(runs_precisions_list)
experiments = len(runs_precisions_list)
lbreak = "\n"

for i in runs_precisions_list:
	precision = i #since indexes to the list start from zero

	date = datetime.now()

	file_path = os.getcwd() + os.sep + "experiments" + os.sep
	results_file = "experiment_"+ str(precision)  + "_" +  date.strftime("%B-%d-%Y") +".txt"
	#print(results_file)
	results  = open(file_path + results_file, "a") 

	batch_size = 128
	num_target_values = 2
	epochs = 1
	steps_per_epoch = 32



	# TODO move paths to command-line arguments
	#image_path = "/media/sf_data_1/bp_aug_aligned"
	image_path = "faces" 
	label_path = "image_to_smile.json"
	patch_size = 128
	# input image dimensions: TODO get from data or command-line params
	input_shape = (patch_size, patch_size, 3)

	train, test, valid = load_data_to_labels(label_path, precision, train_fraction = 0.7, test_fraction = 0.15)

	train_len = len(train)
	test_len = len(test)
	valid_len = len(valid)
	results.write('Input data: train_len: ' + str(train_len) + ", test_len: " + str(test_len) + ", valid_len: " + str(valid_len) + lbreak)

	model = Sequential()
	model.add(SeparableConv2D(32, 3,activation='relu',input_shape=input_shape))
	model.add(SeparableConv2D(64, 3, activation='relu'))
	model.add(Dropout(0.5))
	#model.add(BatchNormalization())
	model.add(MaxPooling2D(2))
	model.add(SeparableConv2D(64, 3, activation='relu'))
	model.add(SeparableConv2D(128, 3, activation='relu'))
	model.add(Dropout(0.5))
	#model.add(BatchNormalization())
	model.add(MaxPooling2D(2))
	model.add(SeparableConv2D(64, 3, activation='relu'))
	model.add(SeparableConv2D(128, 3, activation='relu'))
	model.add(Dropout(0.5))
	#model.add(BatchNormalization())
	model.add(MaxPooling2D(2))
	#model.add(SeparableConv2D(64, 3, activation='relu'))
	#model.add(SeparableConv2D(128, 3, activation='relu'))
	#model.add(MaxPooling2D(2))
	#model.add(SeparableConv2D(64, 3, activation='relu'))
	model.add(GlobalAveragePooling2D())

	model.add(Dense(32, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dense(num_target_values, name="prediction"))

	import keras.backend as K

	def mean_absolute_error(y_true, y_pred):
    		return K.mean(np.absolute(y_true - y_pred))


	model.compile(loss=keras.losses.mean_squared_error,
		      optimizer=keras.optimizers.RMSprop(lr=0.01),
		      # optimizer=keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
	metrics=['mae'])

#if exists("student.mdl"):
#    model.load_weights("student.mdl")

	reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_mean_absolute_error')
	early_stop = keras.callbacks.EarlyStopping(monitor='val_mean_absolute_error', patience=5)
	model_checkpoint = keras.callbacks.ModelCheckpoint("best_student.mdl", save_best_only=True, monitor='val_mean_absolute_error')
	no_nan = keras.callbacks.TerminateOnNaN()

	model.fit_generator(generate_data(image_path, train, batch_size, patch_size),
		    steps_per_epoch=train_len/batch_size,
		    epochs=epochs,
		    callbacks=[reduce_lr,early_stop,model_checkpoint, no_nan],
		    verbose=1,
		    validation_data=generate_data(image_path, valid, batch_size, patch_size),
		    validation_steps=valid_len/batch_size)

	model.save("student.mdl")

	results.write('Evaluation on latest model:' + lbreak)
	score = model.evaluate_generator(generate_data(image_path, test, batch_size, patch_size), steps=16)
	results.write('\tTest loss:' +  str(score[0]) + lbreak)
	results.write('\tTest MSE:' + str(score[1]) + lbreak)

	best_model = keras.models.load_model('best_student.mdl')
	results.write('Evaluation on best model:' + lbreak)
	score = best_model.evaluate_generator(generate_data(image_path, test, batch_size, patch_size), steps=16)
	results.write('\tTest loss:' + str(score[0]) + lbreak)
	results.write('\tTest MSE:' + str(score[1]) + lbreak)



	num_test_samples = 512
	# evaluating on latest model
	results.write("\nCalculating error over " + str(num_test_samples) + " test samples... (using latest model)" + lbreak)
	predictions = model.predict_generator(generate_data(image_path, test, num_test_samples, patch_size), steps=1)
	np.save("predictions_latest.npy", predictions)

	# Make it easy to compare predictions to actuals for test data
	test_vs_predict = []
	for i in range(0, num_test_samples):
    		sample = {"file":test[i][0],  "brightness":test[i][1], "sharpness":test[i][2], "brightness_student":predictions[i][0], "sharpness_student":predictions[i][1]}
    		test_vs_predict.append(sample)

	results.write("Saving errors on " + str(num_test_samples) + " to csv" + lbreak)
	df = pd.DataFrame(test_vs_predict)
	df.to_csv("predictions_vs_test_latest.csv")

	# evaluating on best model
	results.write("\nCalculating error over " + str(num_test_samples) + " test samples... (using best model)" + lbreak)
	predictions = best_model.predict_generator(generate_data(image_path, test, num_test_samples, patch_size), steps=1)
	np.save("predictions_best.npy", predictions)

	# Make it easy to compare predictions to actuals for test data
	test_vs_predict = []
	for i in range(0, num_test_samples):
		sample = {"file":test[i][0], "brightness":test[i][1], "sharpness":test[i][2],
		"brightness_student":predictions[i][0], "sharpness_student":predictions[i][1]}
		test_vs_predict.append(sample)

	results.write("Saving errors on " + str(num_test_samples) + " to csv" + lbreak)
	df = pd.DataFrame(test_vs_predict)
	df.to_csv("predictions_vs_test_best.csv")

	results.close()
