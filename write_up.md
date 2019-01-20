# **Traffic Sign Recognition** 

## Writeup

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

[image1]: ./Write_up_images/visualisation_plot.png "Visualization"
[image2]: ./Write_up_images/train_histogram.png "Training Set Histogram"
[image3]: ./Write_up_images/valid_histogram.png "Validation Set Histogram"
[image4]: ./Write_up_images/Grayscaling.png "Grayscaling Before and After"
[image5]: ./Write_up_images/Translating.png "Translating"
[image6]: ./Write_up_images/Zoom.png "Zooming"
[image7]: ./Write_up_images/Rotate.png "Rotate"
[image8]: ./Write_up_images/tilt.png "Tilt"
[image9]: ./Write_up_images/new_histogram_train.png "Train"
[image10]: ./Write_up_images/new_histogram_valid.png "Valid"
[image11]: ./Internet_Images/30kmhSign.jpeg "30km"
[image12]: ./Internet_Images/BicycleCrossing.jpeg "Bicycle"
[image13]: ./Internet_Images/ChildrenCrossing.jpeg "Children"
[image14]: ./Internet_Images/RoadWorks.jpeg "Roadworks"
[image15]: ./Internet_Images/StopSign.jpeg "Stop Sign"
[image16]: ./Write_up_images/Image_Predictions.png "Softmax Predictions"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library (shape function) to calculate summary statistics of the traffic data set.
Data set summary:

* The size of training set is **34799**
* The size of the validation set is **4410**
* The size of test set is **12630**
* The shape of a traffic sign image is **32x32x3**
* The number of unique classes/labels in the data set is **43**

#### 2. Include an exploratory visualization of the dataset.
I plotted a random image of each class using matplotlib functions. I've also plotted a histogram of the training and validation set. The results are shown below.  

![alt text][image1]
![alt text][image2]
![alt text][image3]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)
##### Grayscaling  
As a first step, I decided to convert the images to grayscale because it reduces the complexity of the image. With 3 RGB channels, a more complex model architecture will be needed. Since the signs can be distinguished just by features alone, a grayscale image is preferrable.  

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image4]

##### Normalising  
As a last step, I normalized the image data because it ensures that the data is on the same scale.    
**Before**    
Mean of train set: 82.677589037  
Mean of valid set: 83.5564273756  
**After**  
Normalized mean of train set: -0.354081335648  
Normalized mean of valid set: -0.347215411128  

##### Resampling
I decided to generate additional data because when I played around with different architectures, I couldn't achieve any higher accuracy than 0.93. I researched ways to improve the accuracy, and resampling was one of the ways. When I looked at the training set histogram, I saw that some classes were underepresented (or too low). Hence, I generated additional data by creating altered copies of the current samples.  

To add more data to the the data set, I altered existing images by tilting, zooming, rotating and translating. This was done by using OpenCV functions.  
**Translating**  
![alt text][image5]
**Zooming**  
![alt text][image6]
**Rotating**  
![alt text][image7]
**Tilting**  
![alt text][image8]

##### New Data Set
If the samples for a particular class were less than 250, new samples for that class would be generated until the number is 4 times as much.  
The samples with fewer than 500 samples were multiplied by 2.  
Other samples sizes remained the same.  
  
New Training Histogram  
![alt text][image9]
New Validation Histogram  
![alt text][image10]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| 1. Convolution    	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 		            |
| 2. Convolution        | 1x1 strides, same padding, outputs 10x10x16   |
| RELU					|												|
| Max pooling			| 2x2 stride, outputs 5x5x16					|
| 3. Convolution        | 1x1 strides, same padding, outputs 1x1x400    |
| RELU					|												|
| Drop out  			| Prob = 0.5                					|
| Add conv2 and conv3 outputs | Output 800								|
| Fully connected		| Output 200        							|
| Fully connected		| Output 43         							|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

| Parameter         	|     Value	        					        | 
|:---------------------:|:---------------------------------------------:| 
| Learning rate         | 0.0009   							            | 
| Epochs    	        | 25 	                                        |
| Patch Size			| 150											|
| Mean	      	        | 0                         		            |
| Std. dev.             | 0.1                                           |


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* Validation set accuracy of **0.969**
* Test set accuracy of **0.921**

If an iterative approach was chosen:
* At first I tried the suggested LeNet structure and it a gave decent result of 0.890 accuracy.
* The accuracy wasn't sufficient for this project, hence I tried adding an additional convolutional layer, and an additional fully connected layer. The accuracy of this model was around 0.92.
* I then decided to try out the suggested architecture provided in the [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). I copied the model archtecture displayed by the figure 2 and it gave an accuracy of somewhere close to 0.93. I tried changing different parameters, like changing batch size and the number of epochs, but it couldn't surpass the 0.93 threshold. Hence, I decided to add more samples to the training and validation set. This improved the accuracy significantly to where it is now.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web (cropped and resized to 32x32 px):

![alt text][image11] ![alt text][image12] ![alt text][image13] 
![alt text][image14] ![alt text][image15]

Initially I cropped the images so that sign occupied the whole image. However, it gave a poor performance of 0.4 accuracy. Hence, I had to recrop the images so that there is some margin on the sides. To ensure that the model can classify the signs correctly, next time more zoomed in images should be generated to prevent this issue.  

The road works sign seems to be hard for the model to classify. It could be due to a patch of dirt on it which isn't visible (now that it has been resized). It also appears to me that the images are having some noise on it due to the jpeg format, and the quality of the image seems to be worse when compared to the provided data set.  

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 30 km/h      		    | 30 km/h   									| 
| Stop Sign     		| Stop Sign 									|
| Children Crossing		| Children Crossing								|
| Bicycle Crossing	    | Bicycle Crossing  			 				|
| Road works			| Right-of-way at the next intersection         |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This is poor compared to the tested accuracy of 96.9%. This could be due to the image quality.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 2nd last code block of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

Image 1: 30 km/h  

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| 30 km/h   									| 
| .00     				| 50 km/h 										|
| .00					| 80 km/h							    		|
| .00	      			| 20 km/h					 			    	|
| .00				    | End of all speed and passing limits     		|
  
Image 2: Stop Sign 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Stop Sign   									| 
| .00     				| Keep Right 									|
| .00					| 70 km/h										|
| .00	      			| 20 km/h					 				    |
| .00				    | No Entry          							|

Image 3: Children Crossing

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Children Crossing 							| 
| .00     				| Dangerous curve to the right 					|
| .00					| Bicycle Crossing								|
| .00	      			| Slippery Road					 				|
| .00				    | Pedestrians        							|

Image 4: Bicycle Crossing    

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Bicycle Crossing   							| 
| .00     				| Children Crossing     						|
| .00					| Road narrows on the right	    				|
| .00	      			| 60 km/h   					 				|
| .00				    | Slippery Road      							|

Image 5: Road works  

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Right-of-way at the next intersection   		| 
| .01     				| Dangerous curve to the right 					|
| .00					| Beware of ice/snow							|
| .00	      			| Road works					 				|
| .00				    | Pedestrians       							|


Displayed results:  
![alt text][image16]

The prediction for "Road works" sign is completely incorrect.  