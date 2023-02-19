# CIFAR-10-Image-Classification

This project aims to classify images from the CIFAR-10 dataset using deep learning techniques. The dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The goal is to classify each image into its corresponding class.

# Dataset
The CIFAR-10 dataset can be downloaded from the official website, or through the Tensorflow and Keras librares. It is split into a training set of 50,000 images and a test set of 10,000 images. Each image is a 32x32 pixel RGB image, with 3 color channels.

# Model
In this project, a convolutional neural network (CNN) is used to classify the images. The model consists of 4 convolutional layers, followed by 2 fully connected layers. ReLU activation function is used for all the layers except the last layer, which uses a softmax function to produce the output probabilities for each class.

The model was trained on the training set for 50 epochs, with a batch size of 64 and a learning rate of 0.001. The training process achieved an accuracy of 79% on the test set.

# Usage
To train the model, run the train.py script. The trained model will be saved in the model.pth file. To evaluate the model on the test set, run the evaluate.py script. The accuracy of the model on the test set will be printed to the console.

# Requirements
Python 3.x
Tensorflow
Keras
NumPy

#License
This project is licensed under the MIT License - see the LICENSE.md file for details.

#Acknowledgments
The CIFAR-10 dataset is provided by the Canadian Institute for Advanced Research (CIFAR).
The model architecture is inspired by the VGG network proposed by Karen Simonyan and Andrew Zisserman in their paper "Very Deep Convolutional Networks for Large-Scale Image Recognition" (https://arxiv.org/abs/1409.1556).


