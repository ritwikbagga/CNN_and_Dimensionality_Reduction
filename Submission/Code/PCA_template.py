# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

#Load data
X = np.load('../../Data/X_train.npy')
Y = np.load('../../Data/y_train.npy')

plt.imshow((X[500]).reshape(28,28))
print(Y[500])
plt.show()
#print(Y[0])

#%% Plotting mean of the whole dataset
# plt.title("Data mean")
# m = np.mean(X.T,axis=1).reshape(28,28)
# plt.imshow( m )
# plt.show()
# print("pass")
#%% Plotting each digit

#%% Center the data (subtract the mean)

#%% Calculate Covariate Matrix

#%% Calculate eigen values and vectors

#%% Plot eigen values

#%% Plot 5 first eigen vectors

#%% Project to two first bases

#%% Plotting the projected data as scatter plot
