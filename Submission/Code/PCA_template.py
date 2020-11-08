# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

#Load data
X = np.load('../../Data/X_train.npy')
Y = np.load('../../Data/y_train.npy')




#%% Plotting mean of the whole dataset
plt.title("Data mean")
mean_X = np.mean(X.T,axis=1).reshape(28,28)
plt.imshow( mean_X)
# plt.show()

#%% Plotting each digit
digits= [0,1,2,3,4,5,6,7,8,9]
digit_means=[]
for digit in digits:
    digit_mean =  np.mean(X[np.where(Y==digit)].T, axis=1).reshape(28, 28)
    digit_means.append(  digit_mean )
fig , axs = plt.subplots(nrows=2, ncols=5, figsize=(10,10))
axs = axs.flatten()
fig.suptitle("Plot Each Digit")
for index , value in enumerate(axs):
    value.imshow(digit_means[index])
    value.set_xticks([])
    value.set_yticks([])
    value.set_title(str(index))
# plt.show()
#%% Center the data (subtract the mean)
X_centered = X-mean_X #centered


#%% Calculate Covariate Matrix
cov_m = np.cov(X_centered.T)
#%% Calculate eigen values and vectors

#%% Plot eigen values

#%% Plot 5 first eigen vectors

#%% Project to two first bases

#%% Plotting the projected data as scatter plot
