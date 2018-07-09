# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup/cnn-architecture.png "Model Visualization"
[image2]: ./writeup/udacity_histogram.png "Histogram of Udacity Data"
[image3]: ./writeup/udacity_histogram_subsampled.png "Histogram of Udacity Data after subsampling"
[image4]: ./writeup/corner.jpg "Cornering"
[image5]: ./writeup/recovery1.jpg "Recovery - Before"
[image6]: ./writeup/recovery2.jpg "Recovery - After"
[image7]: ./writeup/training.png "Training"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is a convolutional neural network based on the NVidia self driving car deep learning model shown below.

![alt text][image1]

My model includes 5 convolutional layers and 3 dense layers with RELU activation functions to introduce non-linearity and batch normalization between layers to better regularize the data and reduce overfitting. (model.py lines 71-112) 

#### 2. Attempts to reduce overfitting in the model

I tried both dropout and batch normalization to reduce overfitting, however I got better results with batch normalization and found it difficult to train the model with both dropout and batch normalization layers.  After doing some research on this topic I found that batch normalization can regularize a model and reduce the need for dropout (https://arxiv.org/pdf/1502.03167.pdf), and that using batch normalization and dropout together can often cause issues (https://arxiv.org/pdf/1801.05134.pdf). (model.py lines 71-112)

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 63). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 112).

#### 4. Appropriate training data

I used a combination of the data provided by Udacity, along with data I collected myself for training the model.  The data provided by Udacity was sufficient to model center lane driving but I needed to add additional data to better model driving around sharp corners and recovering from the sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to to use deep learning to classify driving images using the steering angle as a label.

My first step was to use a convolution neural network model similar to the NVidia self driving car deep learning model documented in NVidia's self driving car blog (https://devblogs.nvidia.com/deep-learning-self-driving-cars/).  I thought this model might be appropriate because it's a model that has been successfully applied to self driving cars and proven to work.

To combat the overfitting, I experimented with adding batch normalization and dropout layers on the NVidia model.  Batch normalization seemed to work better than dropout, and I found it difficult to train a model with both batch normalization and dropout layers. After doing some research on this topic I found that batch normalization can regularize a model and reduce the need for dropout (https://arxiv.org/pdf/1502.03167.pdf), and that using batch normalization and dropout together can often cause issues (https://arxiv.org/pdf/1801.05134.pdf).

After intially training the model on the Udacity sample data I ran it on the simulator to see how well the car was driving around track one.  The model initally captured straight line driving well but struggled with corners, so I modified the dataset to remove excessive straight line driving, and collected more data of the car turning around corners.  Then the model could make it around corners, but would ocassionally drive off the road, so I added data for recovering from the sides.  After a few more iterations of adding data for senarios where the car was struggling/driving off the road, I was able to get the car to drive autonomously around the track.

More details of the data collection and training process is included below.

#### 2. Final Model Architecture

The final model architecture (model.py lines 71-112)) consisted of a convolution neural network with layers as visualized in the table below:

| Layer					|Description									| 
|:---------------------:|:---------------------------------------------:| 
| Input					| 160,320,3 BGR Image							|
| Lambda				| (x / 255.0) - 0.5)							|
| Cropping				| cropping=((70,25),(0,0))						|
| Convolution 5x5		| 24 filters, subsample=(2,2), valid padding	|
| Batch Normalization	| 												|
| RELU					| 												|
| Convolution 5x5		| 36 filters, subsample=(2,2), valid padding	|
| Batch Normalization	| 												|
| RELU					| 												|
| Convolution 5x5		| 48 filters, subsample=(2,2), valid padding	|
| Batch Normalization	| 												|
| RELU					| 												|
| Convolution 3x3		| 64 filters, valid padding						|
| Batch Normalization	| 												|
| RELU					| 												|
| Convolution 3x3		| 64 filters, valid padding						|
| Batch Normalization	| 												|
| RELU					| 												|
| Flatten				| 												|
| Dense					| Outputs 100  									|
| Batch Normalization	| 												|
| RELU					| 												|
| Dense					| Outputs 50  									|
| Batch Normalization	| 												|
| RELU					| 												|
| Dense					| Outputs 10  									|
| Batch Normalization	| 												|
| RELU					| 												|
| Dense					| Outputs 1  									|


#### 3. Creation of the Training Set & Training Process

I started with the the Udacity sample data (udacity_driving_log.csv), which I augmented by horizontal flipping and using the side cameras with a correction factor of 0.2. I split the image and steering angle data into a training and validation set, and made sure the model was training and not overfitting.  After successfully training a model, I tested it out in autonomous mode.  I found the model captured straight line driving well enough but struggled on corners.  After examing the Udacity test data, I found that steering angles were not well distributed in the data set, and that most of the data points had a steering angle of zero.  Below is a histogram of Udacity's sample data:

![alt text][image2]

I modified Udacities sample data by randomly removing 90% of the datapoints where the steering angle was zero (udacity_driving_log_subsample.csv), and the resulting new histogram is shown below:

![alt text][image3]

I trained and tested out the model again on the subsampled dataset, and this time it made it around the first corner, but later drove off the side of the road.  I felt that I needed more data to train the car to drive around sharp corners and to recover before driving off the road, so I collected my own data to add to the Udacity sample data for these situations.

Below is a picture of the car driving around a sharp corner:

![alt text][image4]

And before and after recovering from the side of the road:

![alt text][image5]
![alt text][image6]

After the collection process, I had 3698 number of data points that I collected myself, which I merged with the Udacity sample data to create my final training data set with 11734 data points.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The model seemed to start overfitting slightly after around 15 epochs, however I tried various versions of the model trained over 15, 18, 20, 25, and even 30 epochs and found the test track performance to be best on the model trained on 20 epochs, which is my final model.h5. I used an adam optimizer so that manually training the learning rate wasn't necessary.  A visualization of the training is shown below:

![alt text][image7]
