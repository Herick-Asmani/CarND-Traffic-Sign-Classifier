# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Images/image1.png "visualization train"
[image2]: ./Images/image2.png "visualization valid"
[image3]: ./Images/image3.png "visualization test"
[image4]: ./My_Traffic_signs/1.png "Traffic Sign 1"
[image5]: ./My_Traffic_signs/2.png "Traffic Sign 2"
[image6]: ./My_Traffic_signs/3.png "Traffic Sign 3"
[image7]: ./My_Traffic_signs/4.png "Traffic Sign 4"
[image8]: ./My_Traffic_signs/5.png "Traffic Sign 5"
[image9]: ./My_Traffic_signs/6.png "Traffic Sign 6"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/Herick-Asmani/CarND-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is scattered wrt the labels.
For Training set:

![visualization train][image1]

For Validation set:

![visualization valid][image2]

For Test set:

![visualization test][image3]



### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I didn't convert the images to grayscale but kept it as RGB only because Convolutional Neural Networks works fine with RGB images as with grayscale images.

As a second step, I normalized the image data by dividing by 255 in order to obtain the pixel values between 0.0 and 1.0. Normalization helps in faster learning. Also, Normalization will help to remove distortions caused by lights and shadows in an image.
 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 16x16x16 	|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 16x16x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 8x8x32 	|
| Flatten       		| Output: 2048  								|
| Fully Connected		| Input: 2048, Output: 512						|
| Dropout         		|		0.6										|
| RELU      			|												|
| Fully Connected		| Input: 512, Output: 128						|
| Dropout       		|		0.6										|
| RELU      			|												|
| Fully Connected		| Input: 128, Output: 43						|
| SOFTMAX      			|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used following hyperparameters:
Optimizer: Adam
Batch size: 64
Epochs: 40 (best model obtained at epoch 24)
Learning rate: 0.001



#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100.0
* validation set accuracy of 97 
* test set accuracy of 95.9

Initially, The LeNet-5 architecture was chosen as taught in LeNet Lab. It was chosen because it performed very well on MNIST dataset. So, it was chosen as a baseline model but it didn't perform good on Traffic Sign Dataset.
The model was overfitting and the test set accuracy obtained was low. Actually it was around 92.5% but still it can be improved.
The architecture was modified by increasing the number of filters from 6 to 16 (for first conv layer), 16 to 32 (for second conv layer) compared to initial model. Also both the conv layers were changed to "SAME" padding so that no information is lost during convolution. And the filter size for both convolution were changed from 5x5 to 3x3. Due to change in dimensions of Convolutional layers, Dimensions of Fully Connected layers were also changed. In order to address the problem of overfitting of previous architecture, Dropout with keep probability of 0.6 was used after each Fully Connected layer except for Output layer. In addition to above change, learning rate was changed to 0.001 from 0.0001 which resulted in faster convergence. Batch size was changed from 128 to 64 which slowed the computation little bit but resulted in better accuracy. And a check was added in the code such that if the validation loss was reduced compared to previous epoch's validation loss then save that model and that particular validation loss (so that we can use it for comparing with the upcoming losses and save that model). This way the best model with above mentioned accuracies was obtained.



### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![Traffic Sign 1][image4] ![Traffic Sign 2][image5] ![Traffic Sign 3][image6] 
![Traffic Sign 4][image7] ![Traffic Sign 5][image8] ![Traffic Sign 6][image9]

The third image might be difficult to classify because it included watermark in it. The fourth image might be difficult to classify because it included a German sentence along with the sign.



#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

|     Image			                    |     Prediction	        					| 
|:-------------------------------------:|:---------------------------------------------:| 
| General caution  	                	| 18 'General caution'							| 
| Right-of-way at the next intersection | 11 'Right-of-way at the next intersection'	|
| Road work          					| 25 'Road work'								|
| Go straight or right   	      		| 36 'Go straight or right'		 				|
| Yield                     			| 13 'Yield'          							|
| Speed limit (30km/h)      			| 1 'Speed limit (30km/h)'      				|


The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 95.9%.



#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is relatively sure that this is a General caution sign (probability of 1.0), and the image does contain a General caution sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000      			| General caution								| 
| 0.00000  				| Traffic signals								|
| 0.00000				| Pedestrians									|
| 0.00000     			| Road narrows on the right		 				|
| 0.00000			    | Right-of-way at the next intersection			|


For the second image, the model is relatively sure that this is a Right-of-way at the next intersection sign (probability of 1.0), and the image does contain a Right-of-way at the next intersection sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000      			| Right-of-way at the next intersection			| 
| 0.00000  				| Beware of ice/snow							|
| 0.00000				| Traffic signals								|
| 0.00000     			| Double curve          		 				|
| 0.00000			    | General caution                    			|


For the third image, the model is relatively sure that this is a Road work sign (probability of 1.0), and the image does contain a Road work sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000      			| Road work                         			| 
| 0.00000  				| Go straight or right							|
| 0.00000				| Dangerous curve to the right  				|
| 0.00000     			| Bumpy road             		 				|
| 0.00000			    | Children crossing                   			|


For the fourth image, the model is relatively sure that this is a Go straight or right sign (probability of 0.82614), and the image does contain a Go straight or right sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.82614    			| Go straight or right                 			| 
| 0.09976  				| Bumpy road         							|
| 0.06372				| No entry                      				|
| 0.00637     			| Road work             		 				|
| 0.00191			    | Traffic signals                    			|


For the fifth image, the model is relatively sure that this is a Yield sign (probability of 1.0), and the image does contain a Yield sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000      			| Yield                             			| 
| 0.00000  				| Keep right         							|
| 0.00000				| Turn right ahead              				|
| 0.00000     			| Turn left ahead               				|
| 0.00000			    | Speed limit (50km/h)                          |


For the sixth image, the model is relatively sure that this is a Speed limit (30km/h) sign (probability of 0.99994), and the image does contain a Speed limit (30km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99994    			| Speed limit (30km/h)                 			| 
| 0.00004  				| Priority road        							|
| 0.00001				| End of all speed and passing limits   		|
| 0.00000    			| End of speed limit (80km/h)	 				|
| 0.00000			    | Roundabout mandatory                 			|



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


