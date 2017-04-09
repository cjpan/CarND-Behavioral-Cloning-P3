# **Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./outputs/center_013.jpg "Center Image"
[image2]: ./outputs/left_013.jpg "Recovery Image Left"
[image3]: ./outputs/right_013.jpg "Recovery Image Right"
[image4]: ./outputs/flip.jpeg "Flip Image"
[image5]: ./outputs/shift.jpeg "Shift Image"
[image6]: ./outputs/bright.jpeg "Bright Image"
[image7]: ./outputs/shadow.jpeg "Shadow Image"
[image8]: ./outputs/crop.jpeg "Cropped Image"
[image9]: ./outputs/figure_1.jpg "Error Visualization"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results
* run.mp4 autonomous driving video running on simulator

#### 2. Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I copy almost the full Nvidia's architecture introduced in the lesson and  their [paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).

The only difference from the original network is that I use 2 dropout layers in fully connected layers to reduce overfitting.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 144 and 147).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 14).
The training set and validation set are splitted from the same sample set with 90% for validation.
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 154).
The batch size is set to 128.
I run 10 epochs of training to ensure that the training was enough to make the error decrease to a stable value.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.
I used a combination of center lane driving, recovering from the left and right sides of the road.
So I randomly undersample 20% of the steering data within (-0.1, 0.1) to keep the training set balance.
And I augmented the data with randomly flipping, shifting, adjusting brightness and adding random shadow for more data.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The network is used to map raw pixels from front camera to predict steering commands.

My first step was to use a convolution neural network model similar to the Nvidia's network introduced in the lesson and their [paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). I thought this model might be appropriate because it is used for the similar goal to predict driving commands according to the images captured by camera in pratical situation.

In order to gauge how well the model was working, I split my image and steering angle data into training and validation sets.
My model had a low mean squared error on both the training set and validation set. But in the track testing, the car always failed in the sharp turn. This implied the model was overfitting.

I noticed the data set is inbalanced. The majority of the steering data is very small. So I try undersampling the major data for balance by 20%.
Also I augmented the data with random flipping, shifting, brightness adjustion and random shading to generate randomly new data for training to against the overfitting.

Then the car sometimes would go to one side immediately when it starts after training. I suppose it became overfit to turn in that case. To combat the overfitting, I add 2 dropout layers into the models.

At the end of the process, the vehicle is able to drive autonomously around the track in both directions without leaving the road. It still a bit drives zigzag in the starting straight track, but it would not go off the track. And it drives better in the reverse direction.

#### 2. Final Model Architecture

I clone the training network from Nvidia's paper. It is a working network and created for similar usage, so I suppose it could work as well in my project.

The model consists of a convolution neural network with 5x5 filter sizes and 3x3 filter sizes, and depths between 24 and 64 (model.py lines 138-142)

The model includes RELU layers to introduce nonlinearity (model.py lines 138-142), and the data is normalized in the model using a Keras lambda layer (code line 137).

Fully connected layers follows the convolutional layers.

The only difference from the original network is that I use 2 dropout layers in fully connected layers to reduce overfitting.

Here is my network architecture:

|Layer (type) | Description |
|:--------------------------:|:--------------------------:|
| lambda_1 (Lambda) | 66x200x3 RGB image with normalization |
| conv2d_1 (Conv2D) | 5x5 kernel size, 2x2 stride, output 31x98x24, RELU activation |
| conv2d_2 (Conv2D) | 5x5 kernel size, 2x2 stride, output 14x47x36, RELU activation |
| conv2d_3 (Conv2D) | 5x5 kernel size, 2x2 stride, output 5x22x48, RELU activation |
| conv2d_4 (Conv2D) | 3x3 kernel size, 1x1 stride, output 3x20x64, RELU activation |
| conv2d_5 (Conv2D) | 3x3 kernel size, 1x1 stride,  output 1x18x64, RELU activation |
| flatten_1 (Flatten) | output 1152 |
| dropout_1 (Dropout) | dropout probability 50% |
| dense_1 (Dense) | output 100 |
| dense_2 (Dense) | output 50 |
| dropout_2 (Dropout) | dropout probability 40% |
| dense_3 (Dense) | output 10 |
| dense_4 (Dense) | final output 1 |

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I used the recorded images provided by Udacity. I did not record the track by myself.

The data set provided by Udacity also contains steering data and images captured by center, left and right cameras. The left/right images could train to recover the car from the left side and right sides of the road back to center when it trends to go off the track. Here are examples for center/left/right images for steering of +0.13.

![alt text][image2]
![alt text][image1]
![alt text][image3]

I noticed that the 0-steering angle is over 54% in the whole data set. And in over 74% of the data, steering angle is within (-0.1, 0.1). That means the car would trend to go straight after the training.
So I randomly undersample 20% of the steering data within (-0.1, 0.1) to keep the training set balance.

I splitted the data into training set and validation set by 90% and 10%.
After the collection process, I had 2928 training images and 804 validation images.  

For each frame, I randomly select one of images from the 3 cameras for training.
I add an offset of steering by +-0.25 for the images captured by left and right cameras. For the left camera, the steering is + 0.25, and -0.25 for the right, indicating it needs a larger/smaller steering for recover to the center.

In training set, I randomly flipped half of the images and steerings. That can simulate the case for reverse direction driving.

To augment the data sat, in addition to flip the image, I also randomly shift images and angles, adjust brightness and add random shadows. That would help to generate more new data to fight against overfitting to the "normal" data and improve the network's robustness.

For example, here is an image that has then been processed:

![alt text][image8]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]

The image is resized to 66x200 for input to the network.
I finally randomly shuffled the data set each time appeding data.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 to assure the training was sufficient.
The training error was going down at start and then going to a stable value. I did not augment the validation set, so its error is stable and better than the training error.
![alt text][image9]

I used an adam optimizer so that manually training the learning rate wasn't necessary.
