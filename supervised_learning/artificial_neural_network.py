#!/usr/bin/env python
# coding: utf-8
# Artificial Neural Networks
#
# Artificial Neural Network (ANN) is a type of supervised machine learning. The name and the algorithm both are inspired by the human brain. Like neurons in our brains helps us process information similar to that node in ANN also helps to process information. An ANN aims to solve any specific problem in the same way as a human brain would. ANN consist of multiple nodes which mimic the biological neurons of a human brain. As they are connected through links, they interact by taking the data and performing operations on it and then passing it over to the other connected node.
#
# This unit will talk about a specific type of neural network algorithm called multi-layer perceptrons (MLP).
#
# A multi-layer perceptron classifier (MLP) is a classifier that consists of multiple layers of nodes. Each layer is fully connected to the next layer in the network. Each link between the nodes has a certain weight. The algorithm adjusts the weights automatically based on the previous results. If the results are good then weights are not changed but if the results are not desirable then the algorithm alters the weights.
# Independent and Dependent Variable
# Array X of size (n_samples, n_features), holds the training samples represented as floating-point feature vectors. Array y of size (n_samples) holds the target values (class labels) for the training samples.
from sklearn.neural_network import MLPClassifier
X = [[0., 0.], [1., 1.]]
y = [0, 1]
print("X:", X)
print("y:", y)
# MLP Classifier Model
# We will use the MLPClassifier function from sklearn to initialise the model and save it in the clf variable. Next, we will call the fit method to train the model.
# Import MLPClassifier
# Create the model
clf = MLPClassifier(alpha=1e-05, hidden_layer_sizes=(5, 2),
                    random_state=1, solver='lbfgs')
# Fit the model
clf.fit(X, y)
# Make Prediction
# After fitting (training), the model can predict labels for new samples.
print("Prediction:", clf.predict([[2., 2.], [-1., -2.]]))
# Model Coefficients
# MLP can fit a non-linear model to the training data. clf.coefs_ contains the weight matrices that constitute the model parameters. This is the same as in the Linear Regression Model, where we have betas for every independent variable.
print("Coefficient shapes:", [coef.shape for coef in clf.coefs_])
# Probability Estimation
# Currently, MLP Classifier supports only the Cross-Entropy loss function, allowing probability estimates by running the predict_proba method. MLP trains using Backpropagation. More precisely, it trains using gradient descent, and the gradients are calculated using Backpropagation. For classification, it minimizes the Cross-Entropy loss function, giving a vector of probability estimates per sample.
print("Probability estimates:", clf.predict_proba([[2., 2.], [1., 2.]]))
# We can use the MLP classifiers to train for the particular sets of input variables and use the model to predict the outcomes, where the model uses the appropriate weights of the input layers to predict the outcome. If the results are in line with the training sample, then the weights are not changed. But if the outcome is not optimised, then the model changes each layer's weights and gives the desired result.
