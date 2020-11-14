# CNN and Dimensionality Reduction

# 1) Convolutional Neural Networks
we will use MNIST dataset of handwritten digits containing 10 classes (digits 0-9). Each digit is a 28X28 pixel image, and we will use CNN to classify each one of them. 
The model we have used is made of 2 convolutional layers(activation- ReLu) followed by a maxpool2d layer and then we use a fully connected linear lear. 
The model we have used is made of 2 convolutional layers(activation- ReLu) followed by a maxpool2d layer and then we use two fully connected Linear layers with first linear layer having activation as Re (activation- ReLu) and use a Linear layer again(activation-Softmax).

Results- 
We were able to get a ACC of 94% on training set and 93% on test set 



# 2) Singular Value Decomposition
we have tried optimal k rank approximation and also implemented a sub-optimal k rank approximation to have important computational savgings. 
We have used a picture of a Babboon (X) and tried to reconstruct the images- optimal  and sub-optimal 60 rank approximations using SVD. 
![images](https://github.com/ritwikbagga/CNN_and_Dimensionality_Reduction/blob/master/Figures/Reconstructed_Images.png)

  
  
# 3) Principle Component Analysis (MNIST dataset of handwritten digits)
We will: 
1) Use PCA to get a new set of bases
2) Display the sample mean of the dataset as an image
3) Display the top 5 eigenvectors as images
4) Project your dataset on the first and second principal components 
