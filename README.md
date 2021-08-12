# cnn-handwritten-digit-recognition-with-pytorch
using CNN(with pytorch) and private dataset to recognize handwritten digit.

## Environment
+ python 3.9.6 64-bit
+ numpy 1.19.5
+ matplotlib 3.4.2
+ skimage 0.18.2
+ pytorch 1.9.0+cpu
+ torchvision 0.10.0+cpu

## File Organization
+ dataset/: contains training and testing sets(txt file keep the labels of images) 
+ data_prepare.py: prepare the data pytorch net needs
+ model_cnn.py: define the cnn model, loss function and optimizer
+ train.py: train the model using training set and save the parameters of net
+ test.py: test the model using testing set and get the accuracy the model
+ hand_digit_classifier.ipynb: contains all the python code

## Dataset
+ training set: 1,000 * 10 = 10,000 images (28 * 28)
+ testing set: 500 * 10 = 5,000 images(28 * 28)

## Model & Loss & Optimizer
+ network:
conv1(5 * 5)->relu->pool1(2 * 2)
->conv2(5 * 5)->relu->pool2(2 * 2)
->fc1->fc2
+ loss function: CrossEntropyLoss
+ optimizer: SGD

## Contact Me
+ Email: taoflyfromzero@qq.com
+ Wechat: Flyfromzero
+ Github: Flyfromzero

<br>**Welcome to send me messages and I am glad to help you solve problems.**
