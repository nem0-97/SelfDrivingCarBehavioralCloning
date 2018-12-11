# Behavioral Cloning Project

Overview
---
This project uses a deep convolutional neural network to clone driving behavior. It has been trained, validated and tested using Keras with Tensorflow, the data in this repository, and [this simulator](https://github.com/udacity/self-driving-car-sim). The model will output a steering angle to an autonomous vehicle, the one in the simulator, in order to drive without going off track.

The deep convolutional neural network takes an image (from the car's center camera when driving) and outputs a steering angle.

This repository includes:
* model.py (script used to create and train the model)
* drive.py (script to drive the car)
* model.h5 (the trained Keras model)
* video.py (a script to combine all images in a directory into a mp4 file)
* video.mp4 (a video recording of what the vehicle's center camera saw driving autonomously around the track using the trained model)
* data (a directory containing data from a user driving through the simulator, a CSV file with left, right, and center image paths, steering angle, throttle, braking, and speed, and the images folder containing the image files)

### Dependencies
This lab requires:

* The lab environment can be created with CarND Term1 Starter Kit: [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit) or you can setup your own environment, I do recommend just using this one though.

* [Udacity's simulator](https://github.com/udacity/self-driving-car-sim)


## Details About Scripts In This Directory

### `drive.py`

Usage:
```sh
python drive.py model directory
```
model: path to keras model saved as a .h5 file
directory(optional): directory to create and store images from car's center camera into as it is driven, the image file names will be timestamps of when the image was seen.

```sh
python drive.py model.h5 run1
```
Note: Open the simulator in autonomous mode in order for drive.py to communicate with it.

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection. It will also create a directory named run1 and store images from the car's center camera into as it drives. (just remove run1 from the command to not record images)

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`. Video FPS is 60.

### `model.py`

```sh
python model.py
```

Trains deep convolutional neural network using driving data found in the 'data' folder then saves it as 'model.h5'
