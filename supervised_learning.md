## Linear Regression

Linear Regression was developed under the field of statistics to study the relationship between input and output numerical variables but has been borrowed by machine learning to make predictions based on a linear regression equation.

https://blog.quantinsti.com/gold-price-prediction-using-machine-learning-python/    

https://coinmarketcap.com/currencies/bitcoin/historical-data/  



## Logistic Regression

Linear regression is used to predict values of quantities as a linear function of the input values. When predicting a discrete variable, such as whether a grid of pixel intensities represents 0 or 1, we need to classify the input values. Logistic regression is a simple classification algorithm for learning to make such decisions. It is a model that is used when the dependent variable is categorical. 

## K Nearest Neighbours (KNN)

We will build a simple KNN classifier on trading data (SPY). The goal is to predict next-day direction using lagged returns.

Decision Regions 
A simple 2D plot showing predicted regions for class 0 and 1 over the lagged-return space with test points overlaid.

## Support Vector Machine (SVM)

SVM finds a straight line (in 2D) that separates the classes while **maximizing the margin** to the closest points (support vectors).

We build a simple SVM classifier on trading data (SPY) to predict next-day direction using lagged returns.

- **A 2D plot showing predicted regions for class 0 and 1 over the lagged-return space with test points overlaid.**

- **SVM hyperplane (toy demo)**

**Purpose:** illustrate what a linear SVM learns — one separating line and two margin lines.

**Read it:**

- Solid line = **decision boundary** where the SVM score = 0.
- Dashed lines = **margins** at score = ±1.
- **Support vectors** lie on or inside the margins and determine the boundary.
- The margin width reflects the trade-off set by **C** (lower C → wider margin, higher C → tighter fit).

This is a geometry demo only. In the SPY section we keep the same 2-feature pipeline and report **test accuracy** and the **confusion matrix**; the toy plot builds intuition for what the SVM is doing.

## Random Forest

Random Forest, also called Random Decision Forest, is a method in machine learning capable of performing both regression and classification tasks. It is a type of ensemble learning that uses multiple learning algorithms for prediction.

## Artificial Neural Network (ANN)

A multi-layer perceptron classifier (MLP) is a classifier that consists of multiple layers of nodes. Each layer is fully connected to the next layer in the network. Each link between the nodes has a certain weight. The algorithm adjusts the weights automatically based on the previous results. If the results are good then weights are not changed but if the results are not desirable then the algorithm alters the weights. Check this course on Quantra to learn more about <a href="https://quantra.quantinsti.com/course/neural-networks-deep-learning-trading-ernest-chan" target="_blank"> Neural networks</a> and how to use them for trading. 

https://www.ibm.com/think/topics/supervised-learning  

