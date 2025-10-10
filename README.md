# Machine-Learning-for-Trading

#### I'm taking a course from Quantra by QuantInsti: this are my notes and TESTING

https://github.com/QuantInsti  

https://docs.astral.sh/uv/guides/projects/  

https://marketplace.visualstudio.com/items?itemName=ms-python.autopep8 



#### Simple Explanation of Independent and Dependent Variables in Machine Learning

Imagine you're trying to predict house prices. You look at factors like the house's size, number of bedrooms, and location to guess the price. In machine learning, this is like building a model that learns patterns from data to make predictions.

- **Independent Variables (also called Features or Inputs):** These are the "clues" or pieces of information you feed into the model. They're the things that might influence the outcome, but you don't know how they'll affect it yet—the model figures that out.
  Example: In the house price scenario, size, bedrooms, and location are independent variables.
- **Dependent Variable (also called Target or Output):** This is what you're trying to predict or explain. It's the "answer" the model outputs based on the independent variables.
  Example: The house price itself is the dependent variable—the model uses the features to predict this number.

In short: Independent variables = what you know/measure; Dependent variable = what you want to predict. This setup is the foundation for supervised learning, where the model trains on examples of both to learn the relationship!



### Install

uv init  
uv run main.py  
uv add numpy  
uv add pandas  
uv add scikit-learn  
uv run linear_regression.py  
