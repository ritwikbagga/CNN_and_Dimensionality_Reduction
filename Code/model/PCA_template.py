# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

#Load data
X = np.load('../../Data/X_train.npy')
Y = np.load('../../Data/y_train.npy')

#%% Plotting mean of the whole dataset

mean_X = np.mean(X.T,axis=1)
plt.title("Data mean")
mean_X_plot = mean_X.reshape((28,28))
plt.imshow( mean_X_plot)
#plt.show()
plt.savefig('../../Figures/Q3.2_Mean.png')

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
#plt.show()
plt.savefig('../../Figures/Q3.2_Each_Digit.png')
#%% Center the data (subtract the mean)

X_centered = X - mean_X #centered
#%% Calculate Covariate Matrix
cov_m = np.cov(X_centered.T)
#%% Calculate eigen values and vectors
val, vec = np.linalg.eig(cov_m)
#%% Plot eigen values

plt.title("Plot for Eigen Values")
plt.plot(np.real(val))
#plt.show()
plt.savefig('../../Figures/PLOT_EIGEN_VALUES.png')
#%% Plot 5 first eigen vectors
fig, ax = plt.subplots(5,1,figsize=(10,3))
fig.suptitle("5 first eigen vectors")
counter = 0
for x, a in enumerate(ax.flat):
    a.imshow(np.real(vec.T[x]).reshape(28,28))
#plt.show()
plt.savefig("../../Figures/Q3.3_5_first_eigen_vectors.png")

#%% Project to two first bases
projected = np.real(vec.T[:2].dot(X_centered.T))


#%% Plotting the projected data as scatter plot
plt.figure(figsize=(12,8))
temp_l = [0,1,2,3,4,5,6,7,8,9]

for i in temp_l:
    plt.scatter(projected[0][np.where(Y==i)],projected[1][np.where(Y==i)],s=10,label=i)
plt.title("scatter plot")
plt.ylabel("principal component 1")
plt.xlabel("principal component 2")
plt.legend()
#plt.show()
plt.savefig('../../Figures/Q3.4_scatterPlot.png')

