
#################################################################################
# This python script retrains the trained DNN with measurement data
# (i.e. transfer learning happens here)

# After transfer-learning is finished, the predictions are compared with
# the previous model predictions and measurement data.

# Tunability and loss tangent predictions are provided for further test cases
# after training with measurement data

# Final trained model here is used to find the best figure of merit values
# under different frequencies, barium proportions and defect levels (at 20kV/cm)

# --Achintha Ihalage--
# --QMUL--
# 07/2019
#################################################################################


import os
import pandas as pd 
import numpy as np
np.set_printoptions(threshold=np.nan)
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import keras
# from keras.layers import Input, Dense
# from keras.models import Model
import keras.backend as K
pd.set_option('display.max_columns', 100)

DATASET_PATH = "../data/simulated_data.csv"

def load_tunability_data(dataset_path):

	return pd.read_csv(dataset_path)


tunability_data = load_tunability_data(DATASET_PATH)

#############for testing purposes################################################
# test the tunability and loss predictions before training with measurement data
# test cases can be test1, test2, ..., test6
TEST_PATH1 = '../data/verification/test1.csv'
TEST_PATH2 = '../data/verification/test2.csv'
TEST_PATH3 = '../data/verification/test3.csv'
TEST_PATH6 = '../data/verification/test6.csv'
#################################################################################

#denormalization if required - not required in the current model
def minmax_denormalize(arr):
	max_val = max(np.array(tunability_data['loss']))
	min_val = min(np.array(tunability_data['loss']))

	denorm_arr = arr * (max_val - min_val) + min_val
	return denorm_arr


###########################for testing purposes#############################
test_data6 = load_tunability_data(TEST_PATH6)
test_in6 = test_data6[['1k', '10k', '100k', '1M', '10M', '100M', '1G', '10G', 'electric field', 'defect', 'x', 'Mg']]


## If required to test the tunability and loss tangent predictions before training with measurement data
## uncomment the following snippet and observe the predictions for different test cases 
## (test_case6 is given as an example)
'''
simulation_prediction = load_model('saved_model/bst_mg_dnn.h5').predict(np.array(test_in6))
print simulation_prediction
predicted_tun = simulation_prediction[0]
predicted_loss = simulation_prediction[1]/100
print predicted_loss
print test_data6['tunability']
print predicted_loss
plt.plot(predicted_tun,label='prediction')
plt.plot(test_data['tunability'],label='ground truth')
plt.xlabel('Electric Field')
plt.ylabel('Loss Tangent')
plt.legend()
plt.show()
'''
##############################################################################


# load the training measurement data
MEASURED_TRAIN_DATA='../data/measurement_data.csv'
measured_data = load_tunability_data(MEASURED_TRAIN_DATA)
measured_data['loss'] = measured_data['loss']*100.0
# print measured_data.info()

train_set, test_set = train_test_split(measured_data, test_size=0.1, random_state=8)

train_xs = train_set[['1k', '10k', '100k', '1M', '10M', '100M', '1G', '10G', 'electric field', 'defect', 'x', 'Mg']]
train_ys = train_set[['tunability', 'loss']]

# load the test data that are not present in training set
MEASURED_TEST_DATA = '../data/measurement_test_data.csv'
measured_test_data = load_tunability_data(MEASURED_TEST_DATA)
measurement_test_set = measured_test_data[['1k', '10k', '100k', '1M', '10M', '100M', '1G', '10G', 'electric field', 'defect', 'x', 'Mg']]

################ For figure of merit optimization ###############
fom_data = load_tunability_data('FOM_ML/best_fom.csv')
fom_data = fom_data[['1k', '10k', '100k', '1M', '10M', '100M', '1G', '10G', 'electric field', 'defect', 'x', 'Mg']]
print fom_data.info()
#################################################################



# if R^2 is required as a metric
def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )



##########retrainin with the measurement data happens here############
# I.e. transfer learning
def retrain(learn_rate, batch_size):

	try:
		model = load_model('saved_model/bst_mg_dnn.h5')
		for layer in model.layers:
			# layer.trainable = False

			if layer.name=='loss_dense3' or layer.name=='loss_out': #only trainable two layers
				layer.trainable = True
			else: #freeze these layers
				layer.trainable = False
				print layer.name

		for i,layer in enumerate(model.layers):
			print i,layer.name,layer.trainable
		adam = keras.optimizers.Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.000001, amsgrad=False)
		model.compile(optimizer=adam, 
		              loss='mean_squared_error',
		              metrics=['mse'])
		print model.summary()

		filepath='saved_model/bst_mg_run_best.h5'
		checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)
		
		model.fit(np.array(train_xs), [np.array(train_ys['tunability']),np.array(train_ys['loss'])], batch_size=batch_size, validation_split=0.1, epochs=200, callbacks=[checkpointer])
		model.save('saved_model/bst_mg_run.h5')


	except KeyboardInterrupt:
		print 'Not completed'
		model.save('saved_model/bst_mg_run.h5')


	#######################################################################################################################################################################################################
	# predictions before training with measurement data
	# therefore used bst_mg_dnn_trained.h5 file
	no_mes_train_pred = load_model('saved_model/bst_mg_dnn_trained.h5').predict(np.array(measurement_test_set))

	# predictions after training with measurement data
	# therefore use bst_mg_dnn_trained.h5 file
	measurement_prediction = load_model('saved_model/bst_mg_run_trained.h5').predict(np.array(measurement_test_set))

	# measurement value of the loss tangent from the test set (for comparison)
	loss_measurement = np.array(measured_test_data['loss'])


	# print the loss tangent predictions before and after training with measurement data (i.e. before and after transfer learning)
	# scale down the loss tangent prediction by a factor of 100
	compare_df = pd.DataFrame({'Before transfer-learning':no_mes_train_pred[1].flatten()/100,'After transfer-learning':measurement_prediction[1].flatten()/100, 'Measured loss tangent':loss_measurement}, 
					index=[i for i in range(len(measurement_prediction[0]))], columns= ['Before transfer-learning', 'After transfer-learning', 'Measured loss tangent'])
	print ('\n \n')
	print (compare_df)
	#########################################################################################################################################################################################################


	
	#################################### Tunability and loss predictions for test cases after transfer learning ##################
	# prediction for the test cases (test1, test2, etc.)
	test_data1 = load_tunability_data(TEST_PATH1)
	test_in1 = test_data1[['1k', '10k', '100k', '1M', '10M', '100M', '1G', '10G', 'electric field', 'defect', 'x', 'Mg']]

	test_data3 = load_tunability_data(TEST_PATH3)
	test_in3 = test_data3[['1k', '10k', '100k', '1M', '10M', '100M', '1G', '10G', 'electric field', 'defect', 'x', 'Mg']]
	
	mes_pred1 = load_model('saved_model/bst_mg_run_trained.h5').predict(np.array(test_in1))
	mes_pred3 = load_model('saved_model/bst_mg_run_trained.h5').predict(np.array(test_in3))
	mes_pred6 = load_model('saved_model/bst_mg_run_trained.h5').predict(np.array(test_in6))


	# Plotting happens here
	##### plot the tunability for a test case after training with measurement data
	plt.plot(mes_pred1[0], label='f=1kHz / x=0.5 / defect=0.34 / Mg=0.48')
	plt.plot(mes_pred3[0], label='f=1MHz / x=0.63 / defect=0.25 / Mg=0.4')
	plt.plot(mes_pred6[0], label='f=10GHz / x=0.6 / defect=0.27 / Mg=0')
	plt.legend()
	plt.title('Tunability prediction after training with measurement data')
	plt.xlabel('Electric Field (kV/cm)')
	plt.ylabel('Tunability')
	plt.show()

	##### plot the loss tangent for a test case after training with measurement data
	plt.plot(mes_pred1[1]/100.0, label='f=1kHz / x=0.5 / defect=0.34 / Mg=0.48') # nullify the scalling effect before plotting
	plt.plot(mes_pred3[1]/100.0, label='f=1MHz / x=0.63 / defect=0.25 / Mg=0.4')
	plt.plot(mes_pred6[1]/100.0, label='f=10GHz / x=0.6 / defect=0.27 / Mg=0')
	plt.legend()
	plt.title('Loss tangent prediction after training with measurement data')
	plt.xlabel('Electric Field (kV/cm)')
	plt.ylabel('Loss Tangent')
	plt.show()
	################################################################################################################################


	# #######predictions made to calculated FOM to do optimization on BST material####
	# fom_prediction = load_model('saved_model/bst_mg_run_trained.h5').predict(np.array(fom_data))
	# fom_arr = np.c_[fom_prediction[0], fom_prediction[1]/100]
	# np.savetxt("fom_arr.csv", fom_arr, delimiter=",")


if __name__ == "__main__":

	retrain(learn_rate=0.006,
		  batch_size=5)