import os
import pandas as pd 
import numpy as np
np.set_printoptions(threshold=np.nan)
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import load_model
import matplotlib.pyplot as plt
import keras
# from keras.layers import Input, Dense
# from keras.models import Model
import keras.backend as K


DATASET_PATH = "../data/simulated_data.csv"

def load_tunability_data(dataset_path):

	return pd.read_csv(dataset_path)


tunability_data = load_tunability_data(DATASET_PATH)

#############for testing purposes##############
TEST_PATH = '../data/verification/test6.csv'
###############################################

#denormalization if required/not required in the current model
def minmax_denormalize(arr):
	max_val = max(np.array(tunability_data['loss']))
	min_val = min(np.array(tunability_data['loss']))

	denorm_arr = arr * (max_val - min_val) + min_val
	return denorm_arr


###########################for testing purposes#############################
test_data = load_tunability_data(TEST_PATH)
test_in = test_data[['1k', '10k', '100k', '1M', '10M', '100M', '1G', '10G', 'electric field', 'defect', 'x', 'Mg']]


simulation_prediction = load_model('saved_model/bst_mg_dnn.h5').predict(np.array(test_in))
print simulation_prediction
predicted_tun = simulation_prediction[0]
predicted_loss = simulation_prediction[1]/100
print predicted_loss
print test_data['tunability']
# print predicted_loss
plt.plot(predicted_tun,label='prediction')
plt.plot(test_data['tunability'],label='ground truth')
plt.xlabel('Electric Field')
plt.ylabel('Loss Tangent')
plt.legend()
# plt.show()
##############################################################################


MEASURED_TRAIN_DATA='../data/measurement_data.csv'
measured_data = load_tunability_data(MEASURED_TRAIN_DATA)
measured_data['loss'] = measured_data['loss']*100.0
# pure_bst = pure_bst.dropna(subset=['Measured Tand','Measured Tunability'])
# print measured_data.info()

train_set, test_set = train_test_split(measured_data, test_size=0.1, random_state=8)

train_xs = train_set[['1k', '10k', '100k', '1M', '10M', '100M', '1G', '10G', 'electric field', 'defect', 'x', 'Mg']]
train_ys = train_set[['tunability', 'loss']]

# test_xs = test_set[['10k', '100k', '1M', '10M', '100M', '1G', '10G', 'electric field', 'x', 'defect parameter']]
# test_ys = test_set[['Measured Tand','Measured Tunability']]

MEASURED_TEST_DATA = '../data/measurement_test_data.csv'
measured_test_data = load_tunability_data(MEASURED_TEST_DATA)
measurement_test_set = measured_test_data[['1k', '10k', '100k', '1M', '10M', '100M', '1G', '10G', 'electric field', 'defect', 'x', 'Mg']]


fom_data = load_tunability_data('FOM_ML/best_fom.csv')
fom_data = fom_data[['1k', '10k', '100k', '1M', '10M', '100M', '1G', '10G', 'electric field', 'defect', 'x', 'Mg']]
print fom_data.info()

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
		model.fit(np.array(train_xs), [np.array(train_ys['tunability']),np.array(train_ys['loss'])], batch_size=batch_size, validation_split=0.1, epochs=10)
		model.save('saved_model/bst_mg_run.h5')


	except KeyboardInterrupt:
		print 'Not completed'
		model.save('saved_model/bst_mg_run.h5')

	# predictions before training with measurement data
	no_mes_train_pred = load_model('saved_model/bst_mg_dnn.h5').predict(np.array(measurement_test_set))

	measurement_prediction = load_model('saved_model/bst_mg_run.h5').predict(np.array(measurement_test_set))

	# prediction for the test cases (test1, test2, etc.)
	pr = load_model('saved_model/bst_mg_run.h5').predict(np.array(test_in))
	


	#######predictions made to calculated FOM to do optimization on BST material####
	fom_prediction = load_model('saved_model/bst_mg_run.h5').predict(np.array(fom_data))
	fom_arr = np.c_[fom_prediction[0], fom_prediction[1]/100]
	np.savetxt("fom_arr.csv", fom_arr, delimiter=",")


if __name__ == "__main__":

	retrain(learn_rate=0.0006,
		  batch_size=5)