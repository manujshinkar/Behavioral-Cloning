# Behaviorial Cloning Project

Overview
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road

This project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

Model Architecture
---
In this project I have used the model architecture from Nvidia for training self-driving car. I used this model as a base and made slight changes for my project. The Nvidia architecture is explained in [this](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) paper.

<img src="./Images/nVidia_model.png?raw=true" width="400px">

I started by implementing this architecture as follows:
* Image normalization using a Keras Lambda function so that all pixels fall in range of -1 to 1
* Three 5x5 convolution layers with 2X2 striding
* Two 3x3 convolution layers 
* Three fully-connected layers 
* RELU activation on all convolution and fully connected layers.
* MSE loss and adam optimizer

Initially I started with the udacity provided dataset without augmentation. I trained this model for 10 epochs. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. To combat the overfitting I provided augmented data to the model. Now I observed that the validation loss was reducing for 6 epochs and the oscillating.

I then trained trained the model for 5 epochs. The validation loss is relatively less than the preivous implementation. I tested this model on the simulator. The car drives good, however it tends to go slight outside on the turns.

After this I added L2 regularization on all the convolutional and fully connected layers and changed RELU activations to ELU.
Now the training and validation loss and very less as compared to the previous implementation and the car drives smoothly on the simulator. 

Training Strategy
---

### Data Collection and augmentation
I used the udacity provided data for training. The dataset has images from center, left and right cameras. Around 8000 from each camera.  

Left:
<img src="./Images/left.jpg?raw=true" width="200px">
Center:
<img src="./Images/center.jpg?raw=true" width="200px">
Right:
<img src="./Images/right.jpg?raw=true" width="200px">

Initialy I used the center camera images only to train. The car did not drive well. I then used all the three camera images. For left and right camera I used a steering factor correction factor of 0.25. Then I flipped all the images and measurements to increase my training dataset. Now I have close to 48000 images for training. This was a very helpful step. It worked really well. 

Original:
<img src="./Images/origional.png?raw=true" width="200px">
Flipped:
<img src="./Images/flipped.png?raw=true" width="200px">

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I shuffle the data and use 20% for validation.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5. I used an adam optimizer so that manually training the learning rate wasn’t necessary.

### Data Processing

The images from the simulator contains extra information such as sky and part of the car as we can see in the above camera images. So I cropped the images from top and bottom. 

Original:
<img src="./Images/origional.png?raw=true" width="200px">
Cropped:
<img src="./Images/cropped.png?raw=true" width="200px">

The Nvidia architecture take images of size 200 x 66. So after Cropping the images they are resized. 

The Simulator gives images in RGB format in drive.py and the cv2.imread reads images in BGR in model.py. So in both the files the images are converted to YUV format as suggested in the Nvidia paper.

After the preprocessing the images are fed to the model which performs normalization so it is not included in the preprocessing step in drive.py.



Conclusion
---
In this project, I have learned about deep neural networks and convolutional neural networks to clone driving behavior. I have trained, validated and tested a model using Keras. The model outputs a steering angle to an autonomous vehicle.

I used a simulator where I can steer a car around a track for data collection. I used image data and steering angles to train a neural network and then used this model to drive the car autonomously around the track.


