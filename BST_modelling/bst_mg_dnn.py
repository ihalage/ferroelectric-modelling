
#####################################################################
# This python script implements a split neural network for
# BST material property prediction (tunability and loss tangent)

# Model is trained on the huge database obtained from the simulations

# --Achintha Ihalage--
# --QMUL--
# 07/2019
#####################################################################

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
np.set_printoptions(threshold=np.nan)
from sklearn.utils import shuffle
import keras
# import tensorflow as tf
# import tensorflow.contrib.layers as lays
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
# from keras.layers import Embedding
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import MinMaxScaler
# from keras.layers import LeakyReLU
# import pydot as pyd
# import pydotplus
from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import plot_model



DATASET_PATH = "../data/simulated_data.csv"


def load_tunability_data(dataset_path):
	return pd.read_csv(dataset_path)

# for visualization purposes
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)


tunability_data = load_tunability_data(DATASET_PATH)
print tunability_data.info()

# Normalize the input data if required
def minmax_normalize(arr):
	max_val = max(arr)
	min_val = min(arr)

	norm_arr = (max_val - arr)/(max_val-min_val)
	return norm_arr

# Scale-up the loss tangent by a factor of 100 to bring the loss values closer to the tunability values
# As narrowly spread target values result in better predictions in the regression models
tunability_data['loss'] = tunability_data['loss']*100.0

# Split the dataset into 80% training and 20% testing data
train_set, test_set = train_test_split(tunability_data, test_size=0.2, random_state=8)

print train_set.info()
# scaling if required
scaler = MinMaxScaler()

# After performing one-hot encoding, the training set features look like follows
train_xs = train_set[['1k', '10k', '100k', '1M', '10M', '100M', '1G', '10G', 'electric field', 'defect', 'x', 'Mg']]
train_ys = train_set[['tunability','loss']]
# train_xs = scaler.fit_transform(train_xs)

# print len(np.array(train_xs))

test_xs = test_set[['1k', '10k', '100k', '1M', '10M', '100M', '1G', '10G', 'electric field', 'defect', 'x', 'Mg']]
test_ys = test_set[['tunability','loss']]
# test_xs = scaler.fit_transform(test_xs)

# Number of features after doing one-hot encoding
max_features = 12


###################### Implement the DNN here ##############################
''' 
	Fully connected DNN architecture
		
	f(one-hot), E, x, xi_s, xi_Mg
				 |
			12 neurons
				 |
			50 neurons
			|        |
	100 neurons    100 neurons
		|				|
	100 neurons    100 neurons
		|				|
	50 neurons     50 neurons
		|				|
	1 neuron    	1 neuron
		|				|
	Tunability 		Loss tangent
	prediction	    prediction

'''
features = Input(shape=(12,),name='inputs')
tun_dense1 = Dense(50, activation='elu',name='tun_dense1')(features)
tun_dense2 = Dense(100, activation='elu',name='tun_dense2')(tun_dense1)
tun_dense3 = Dense(100, activation='elu',name='tun_dense3')(tun_dense2)
tun_dense4 = Dense(50, activation='elu',name='tun_dense4')(tun_dense3)
tun_out = Dense(1, activation='elu',name='tun_out')(tun_dense4)

loss_dense1 = Dense(100, activation='elu',name='loss_dense1')(tun_dense1)
loss_dense2 = Dense(100, activation='elu',name='loss_dense2')(loss_dense1)
loss_dense3 = Dense(50, activation='elu',name='loss_dense3')(loss_dense2)
loss_out = Dense(1, activation='elu',name='loss_out')(loss_dense3)

merged_model = Model(inputs=[features],outputs=[tun_out, loss_out])
# usage of keras plot_model to visualize the implemented model
plot_model(merged_model,to_file='merged_model.png',show_shapes=True)

##############################################################################

# R^2 value as a metric if required
# R^2 quantifies the goodness of a fit of a regression model
def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )



def train(learn_rate, batch_size):

	try:
		# AdamOptimizer is used
		adam = keras.optimizers.Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0001, amsgrad=False)

		merged_model.compile(optimizer=adam, 
		              loss='mean_squared_error',
		              metrics=['mse','accuracy'])

		# save the model at every 100 epochs
		checkpoint = keras.callbacks.ModelCheckpoint('saved_model/model{epoch:08d}.h5', period=100) 
		merged_model.fit(np.array(train_xs).reshape(len(train_xs),max_features), [np.array(train_ys['tunability']),np.array(train_ys['loss'])], callbacks=[checkpoint], batch_size=batch_size, validation_split=0.2, epochs=500000)

	# Terminate the training with Ctrl+C when the validation loss settles
	except KeyboardInterrupt:
		print 'Training terminated by the user'
		merged_model.save('saved_model/bst_mg_dnn.h5')



if __name__ == "__main__":

	train(learn_rate=0.0001,
		  batch_size=300)
