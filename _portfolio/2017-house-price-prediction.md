---
title:  "House Price Prediction"
personal: true
tech: Python, Jupyter notebook, XGBoost
methods: >
  Exploratory data analysis, Machine Learning: Lasso regression, Ridge regression, ElasticNet, Gradient Boosting, Support Vector Regression, 
  K-Nearest Neighbour imputation, ensembling
from:  2017-07-01 00:00:00 +0200
to: 2017-08-16 00:00:00 +0200
---

This is a personal project. I competed in [Kaggle's "House Prices: Advanced Regression Techniques"][kaggle] competition partly because I simply liked the challenge, partly with the intent of getting some Python code online as part of my portfolio, but mostly because I wanted to experiment with automating parts of a Machine Learning pipeline. I made it to the top 5% so far.

I started out with an exploratory analysis to understand the data. Armed with this knowledge, I built the following components:

 - a system to filter out ineffective manual data transformations
 - an imputation algorithm to deal with missing data
 - an automatic transformation function that applies further transformations as needed based on heuristics
 
The final predictions were created using an ensemble of multiple models.

See [my blog post][blog_post] for technical details.

The code is on [GitHub][git].


[git]: https://github.com/jvanlier/Kaggle_Houseprices/tree/a1780c2a555de4925df51a4c3db96daeb3fae6a0
[kaggle]: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
[blog_post]: {{ site.baseurl }}{% link _posts/2017-08-26-kaggle-house-price-prediction.md %}
