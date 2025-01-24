# SkinClassification
A custom convolutional architecture which is capable of classifying skin lesions as benign or cancerous, with a test accuracy of 83% when the architecture is used in conjunction with a transfer learning model.

## Skin Cancer Classification Model
Ryan Benner
11/11/24

## Introduction

In this project, I aimed to develop a binary classification model to predict whether pictures of skin lesions are benign or malignant (cancerous). I utilized a dataset that contained images of both benign and malignant lesions that were labeled as such. The goal was to create a model that is capable of aiding anyone in preliminarily diagnosing skin cancer more accurately and efficiently than previously possible. By leveraging convolutional neural networks (CNNs) and transfer learning methods, I aimed to develop proficient classification models while reducing the need for large amounts of computational resources. This work is important in healthcare because it has the potential to enhance medical diagnostics and contribute to better patient outcomes with better classification models being available.

## Analysis

The dataset consisted of high-resolution images of skin lesions with corresponding labels that indicated whether they were malignant or benign. Before training the models, I performed data preprocessing steps. All images were resized to a consistent size of (224, 224) pixels to standardize the input for the neural networks. I normalized the pixel values by dividing by 255.0 to scale them between 0 and 1, which helps stabilize and speed up the training process. The labels were converted into integer format (0 for benign and 1 for malignant) to be prepared for the binary classification task. I also examined the dataset for class imbalance and ended up using class weighting and data augmentation to address the small size of the dataset, and the skewed proportions of benign and malignant images. My data augmentation included various changes, such as rotations and flips, which I did to help increase the diversity of the training data and help the model’s performance metrics improve.

## Methods

I began by constructing a custom convolutional neural network (CNN) model. The architecture included five convolutional blocks with increasing numbers of filters: 32, 64, and 128. Unfortunately, running locally, I was not able to include 256 and 512 filters, as they were too computationally expensive for my machine.  Each block consisted of convolutional layers with ReLU activation, batch normalization, and dropout layers to help with overfitting (was only slightly effective/helpful, considering the small dataset). Max pooling layers were applied after certain blocks to reduce the spatial dimensions. The model ended with a global average pooling layer and dense layers, finishing with a sigmoid activation function for binary classification of the skin lesion pictures. 

To utilize prior knowledge from pre-built models, I implemented transfer learning using the EfficientNetB0 architecture. I loaded the pre-trained EfficientNetB0 model without its top classification layers and froze the weights so it can retain its learned features. Custom top layers were added, including a pooling layer, dropout for regularization, and a final output layer using a sigmoid activation function. I compiled the model using the adam optimizer and binary cross-entropy loss. After the first 10 epochs of training, I unfroze the base model for fine-tuning with a slightly lower learning rate. This allowed the model to train for another 10 epochs and gain an even higher validation accuracy score.

## Results

The custom CNN model achieved a training accuracy of 83.3% and a testing accuracy of 66.7%. The ROC AUC score for the test data was 0.71. These results indicate that the custom model was moderately effective in classifying skin lesions, as a preliminary screening. Initially, the transfer learning model using VGG19 underperformed, with an accuracy of around 0.5, which is equivalent to random guessing. Upon debugging, I found issues such as incorrect preprocessing and not enough top layers, as well as a better performing base model - EfficientNetB0.. After correcting some of the preprocessing steps, enhancing the top layers, and changing + fine-tuning the base model, the transfer learning model’s performance improved significantly. The updated EfficientNetB0 model achieved a training accuracy of 81.9% and a testing accuracy of 83.3%. The ROC AUC score improved to 0.9. These metrics suggest that the transfer learning model, when properly configured, is easily capable of surpassing the performance of the custom CNN. 

## Reflection

This project highlighted the importance of careful data selection and preprocessing, as well as model selection and specific configuration in deep learning scenarios. I saw that even small mistakes, such as double preprocessing or slightly incorrect layer definitions, can drastically affect model performance. Through utilizing transfer learning, I learned about the value of simply fine-tuning pre-trained networks and the impact of adding carefully selected top layers. In the future, I definitely want to make sure I have a larger dataset to work with, and am more careful when beginning the project in the form of evaluating my data and ensuring I preprocess correctly before advancing to the model. This experience showed the need for a lot of experimentation when developing machine learning models, and taught me to be more careful when designing models and engineering data.



