# **Behavioral Cloning**

## Writeup

---
**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report
---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* video.py to combine images from a directory into a video
* model.h5 containing a trained convolution neural network
* writeup.md summarizing the results(this file)

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My final model consisted of the following layers(model.py lines 73-89):

| Layer         		  |     Output        					|
|---------------------|-----------------------------|
| Input         		| 160x320(RGB image)   			|
| Cropping2D         		| 65x320(RGB image)   		|
| Lambda(x=x/255.0-.5)         		| 65x320(normalized image)   	|
| Convolution 5x5,2x2 stride valid padding(w/RELU)| 24 filters|
| Convolution 5x5,2x2 stride valid padding(w/RELU)| 36 filters|
| Convolution 5x5,2x2 stride valid padding(w/RELU)| 48 filters|
| Convolution 3x3,1x1 stride valid padding(w/RELU)| 64 filters|
| Convolution 3x3,1x1 stride valid padding(w/RELU)| 64 filters|
| Flatten(w/ Dropout=.5)				  | |
| Fully connected	(w/ Dropout=.5)	| 100|
| Fully connected	(w/ Dropout=.5)	| 50|
| Fully connected	(w/ Dropout=.5)	| 10 |
| Fully connected (final output)	| 1|

#### 2. Attempts to reduce overfitting in the model

The model I used is based on [the model design in this paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdfcontains)

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 64). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 93).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to have a CNN that could take in an image from the car's camera and output a steering angle.

My first step was to use a convolution neural network model similar to the one used by NVIDIA [in this paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdfcontains) I thought this model might be appropriate because they were able to get good results with it in real life.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.

I forgot the data augmentation for some time(commented it out while experimenting) and ended up only being able to get it to drive the track after 12 epochs without dropout, and no activation on the fully connected layers, I believe the main reason this worked was probably because I had overfit the model to the track.

To combat the overfitting, I modified the model(uncommented the dropout) so that there was a layer of dropout before every fully connected layer with .5 probability of dropping a connection, I also tried to keep the number of epochs as low as possible, and augmented the data by flipping every image passed in and also using left and right images.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track to improve the driving behavior in these cases, I collected more data of it recovering to the center in these areas.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

My final model consisted of the following layers(model.py lines 73-89):

| Layer         		  |     Output        					|
|---------------------|-----------------------------|
| Input         		| 160x320(RGB image)   			|
| Cropping2D         		| 65x320(RGB image)   		|
| Lambda(x=x/255.0-.5)         		| 65x320(normalized image)   	|
| Convolution 5x5,2x2 stride valid padding(w/RELU)| 24 filters|
| Convolution 5x5,2x2 stride valid padding(w/RELU)| 36 filters|
| Convolution 5x5,2x2 stride valid padding(w/RELU)| 48 filters|
| Convolution 3x3,1x1 stride valid padding(w/RELU)| 64 filters|
| Convolution 3x3,1x1 stride valid padding(w/RELU)| 64 filters|
| Flatten(w/ Dropout=.5)				  | |
| Fully connected	(w/ Dropout=.5)	| 100|
| Fully connected	(w/ Dropout=.5)	| 50|
| Fully connected	(w/ Dropout=.5)	| 10 |
| Fully connected (final output)	| 1|

I still plan on adding RELU activation to the fully connected layers, commented it out while testing various models

#### 3. Creation of the Training Set & Training Process

##### NOTE:I did not collect the data as my computer would keep freezing in the simulator as I was trying to drive, even if I made sure that was the only program running, due to this I used the data provided at the beginning of the section. I will discuss how I would go about getting data to try and develop a good robust generalized model

To capture good driving behavior, I would record two laps on track one using center lane driving.

I'd then record the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to move back to the center if it ever veered off course, however make sure not to record you moving the car to the side else the model might pick this up in training These images show what a recovery looks like starting from a non central position in the road, or the side of the road

I would also then repeat this whole process on track two in order to get more data points, and try to make sure the model gets a more general idea of what it means to drive and can potentially navigate more courses and tracks.


#### (model.py lines 34-66):

After the collection process, I had 8,036 number of data points.

To augment the data sat, I also flipped images and angles thinking that this would give more data to train on and avoid it learning to turn in one direction only since the first track is a loop with left turns only.

I also used not just the center image, but also the side images and added .2 to the steering angle for the left image and subtracting .2 for the right, this allows me to add more data as the left and right cameras capture what the car would see from its center camera were it off to the right or left of its current position, and we can extrapolate if it were a little off to the right we would want it to steer back to left a little bit more than if it were where it is, and vice versa for the left.(the .2 is kind of arbitrary, with proper math and measurements of the camera positions you could come up with more exact numbers, or use a guess and check method you might find something that works better)

Augmentation using these techniques gave me 6x as many data points

I then preprocessed this data by using Keras Cropping2d layer as well as a Lambda layer. The cropping takes out parts of the images that are irrelevant, the hood of the car and objects for off in the distance, it removes the top 70 and bottom 25 rows of pixels from the image. The image is then passed through the lambda layer which divides each element by 255 to normalize values between 0 and 1, then subtracts .5 to try and center the data around 0.


I finally randomly shuffled the data set and put 20% of the data into a validation set.

When feeding data to the model for both training and validation I used a generator to load each batch so that the program did not use too much memory, just enough as needed for one batch,(size 32), in this generator is where I also performed my data augmentation, created 3 new flipped images from the left, right, and center images and added 5 new steering values before running the batch though the model.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs seemed to be 2 as it was able to drive well had low loss and was the lowest number of epochs where I had these results(I did not try 1 epoch) I used an adam optimizer so that manually training the learning rate wasn't necessary.
