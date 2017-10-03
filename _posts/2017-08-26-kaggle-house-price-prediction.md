---
layout: post
title:  "Kaggle House Price Prediction"
date:   2017-08-26 15:00:00 +0200
categories: blog
---
I spent some time on [Kaggle's "House Prices: Advanced Regression Techniques"][kaggle_hp] competition the last couple of weeks. Partly because I simply liked the challenge, partly with the intent of getting some Python code online as part of my portfolio, but mostly because I wanted to experiment with automating parts of a Machine Learning pipeline. In this post, I'll share some details of my approach. I managed to get a top 5% score on August 16th, 2017 with a score of 0.11459. 

{: style="text-align: center;"}
![top5](/assets/img/blog/2017-08-26-kaggle-house-price-prediction/top5.png)

I haven't decided yet if I want to keep going - it's kind of a time sink and there's no prize to collect at the end. But hey, there's three more years to go, so there's no rush.

The code is on [GitHub][git]. 
 
# Approach
The competition provides a dataset of 2919 houses, 50% of which is test data. The goal is to predict the sales price for each house house price based on the other 79 features.

I started with an exhaustive exploratory analysis. See [here][git_explo] for the Jupyter Notebook. The dataset is a very small, resulting in lots of freedom to experiment in ways that are usually not practically feasible. That's exactly what I set out to do here: I set up my manual transformations to be optional, such that they can be filtered based on their empirical effectiveness. Only the ones that are effective are applied. Some features now still need work before being ready to be fed into a machine learning algorithm, so the data subsequently gets transformed automatically as needed on a best-effort basis (such as one-hot-encoding). If there's any missing values, they are automaticaly imputed based on similarity to other houses. See [here][git_ml] for the Machine Learning Notebook.

Since I concentrated my efforts on the data preparation steps, I'd like to go over that in detail. The predictions are delivered by a straight forward ensemble of multiple models, which I'll only cover briefly.

## Data preparation

### Manual data transformations
The main goal of my approach was to understand which transformations work and which don't, rather than relying on hunches and intuition. To enable this, I defined my manual transformations as function rather than applying them directly. Each transformation function takes a DataFrame and returns a (transformed) DataFrame. This setup basically makes each transformation optional and makes it possible to quantify their effectiveness. More on that below.
  
### Automatic data transformation
In order to make *optional* manual transformations a possibility, there has to be some kind of fallback operation that handles the features that were not transformed manually. I wrote a function that automatically applies all the required transformations that are needed to get a DataFrame to play well with scikit-learn. This function basically accepts any DataFrame and decides for each numeric feature whether it will scale it or leave it alone and uses label-encoding or one-hot-encoding for categorical features. Code [here][git_autotrans]. 

### Missing data
There are a lot of missing values in this dataset. I dealt with this using an unsupervised *k*-Nearest Neighbors imputer. This imputer fills an average of the *k* closest houses for numeric features and the mode for categorical features. The code can be found [here][git_knn]. *k* was set to 20 in this case. The imputer is invoked after all transformations have been applied, so I could still play with manual imputation to see if something else works better. This setup even permits simply deleting outliers, because a new value will just get imputed (and the new value will be closer to the center of the distribution). 

### Testing the effectiveness of manual transformations
Each manual transformation is tested in turn to determine its effectiveness by running it through the following steps:

- shuffle data
- calculate a baseline score: 
    - apply automatic transforms
    - impute missing data
    - do 10-fold Cross Validation to compute average score
- calculate score with transformation:
    - apply manual transformation under test
    - apply automatic transforms
    - impute missing data
    - do 10-fold Cross Validation to compute average score

I used Scikit-learn's RidgeCV with its inner Leave-One-Out-Cross-Validation loop to find a good value for the regularization parameter for each fold, because otherwise the score varies heavily across folds.

Transformations that do not decrease the loss with respect to the baseline are excluded from the final model building. I simply used 0 as the cutoff here.

Here's a snippet from the output:

```
trans_del_outliers  base: 0.134 (± 0.036) | w/t: 0.115 (± 0.016) | -14.43%
trans_mssubclass    base: 0.134 (± 0.036) | w/t: 0.134 (± 0.036) |  -0.29%
trans_lotshape      base: 0.134 (± 0.036) | w/t: 0.134 (± 0.036) |  -0.15%
trans_landcontour   base: 0.134 (± 0.036) | w/t: 0.135 (± 0.037) |   0.20%
trans_lotconfig     base: 0.134 (± 0.036) | w/t: 0.134 (± 0.036) |   0.13%
trans_landslope     base: 0.134 (± 0.036) | w/t: 0.134 (± 0.036) |   0.09%
trans_neighborhood  base: 0.134 (± 0.036) | w/t: 0.139 (± 0.039) |   3.37%
```

As can be seen here, outlier removal worked very well, but my transformation on neighborhood had a detrimental effect. 

Note that scores vary a little from run to run due to the shuffling, causing a bit of flip-flipping behaviour for the transformations around the 0-point, but the transformations that really matter are consistently included (and the ones that are detrimental are consistently excluded).

A mistake I made initially was *not* shuffling the data between tests. This caused overfitting because the exact same folds were used to make a lot of sequential decisions. Shuffling between tests mitigates that. The second mistake was trying to do this with Boosted Trees. There's too much variability in tree generation with the default parameters, making it impossible to decide whether a transformation is effective or not. And a parameter search takes too much time. A regularized linear model is much more stable, much faster to train and thus better suited for this task.

## Modelling
Training the Machine Learning algorithms is straight forward, so I'll keep this brief. First, all effective manual transformations are applied. Followed by automatic transformations and imputation. I then chucked everything into a 10-fold Cross-Validated Grid Search for a couple of models: Ridge regression, Lasso regression, ElasticNet (which basically finds a combination between Ridge and Lasso), SVR (Support Vector Regressor) with RBF kernel (Radial Basis Function) and - of course - GBT (Gradient Boosted Trees).

Why these models? I expected a regularized linear model to work well, because I spent a lot of effort to make numeric features more Gaussian, to handle outliers and to make features more linearly correlated with SalePrice. I had no expectation about whether L1 or L2 regularization would perform better, hence I ended up just trying both. The SVR was added because I empirically found that it worked very well - no theoretical basis. GBT because, well, they always just work really well.

 An ensemble of Ridge + Elastic + SVR + GBT was used for my highest scoring submission. This performed better than any single model in isolation.
 
# Automatic transformation effectiveness testing: the silver bullet?
Well, no, not really... The transformation effectiveness report has been spot on most of the time, but has on rare occasions been terribly wrong. Some transformations seemed very effective in isolation (10%-15% improvement) but actually degrade leaderboard performance. This might be because transformations are only tested in isolation and the impact of various combinations is unknown, or  because they're only tested on a linear model while the final prediction is ensembled with an SVR and GBTs. Or it's an overfitting issue. More research needed.

# What's next?
How about automatically discovering those manual transformations? Finding those was very time consuming (as evidenced by the length of the `Explorative_Data_Analysis.ipynb` notebook). I've experimented a bit with a randomized transformation generator, which randomly decides to either drop a feature, or combine two features (polynomial), or binarize a categorical feature, or merge a random selection of categories within a categorical feature, etc. So far, it looks promising.

Secondly, manual outlier removal had - by far - the biggest impact on my score. So naturally I'd like to try automatic outlier removal, in order to cover more ground. I already tried a naive univariate version, which didn't work well, but I have good hopes for 2-dimensional version.

The ensemble model selection could also be automated.

# Conclusion
I described how I set up the system with which I've tested my assumptions in Kaggle's "House Prices: Advanced Regression Techniques" challenge. Of course, something like this is only practically feasible with a small data set, but it's been very useful to climb the ranks and it was a lot of fun to build. 


[kaggle_hp]: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
[git]: https://github.com/jvanlier/Kaggle_Houseprices/tree/a1780c2a555de4925df51a4c3db96daeb3fae6a0
[git_explo]: https://github.com/jvanlier/Kaggle_Houseprices/blob/a1780c2a555de4925df51a4c3db96daeb3fae6a0/Explorative_Data_Analysis.ipynb
[git_ml]: https://github.com/jvanlier/Kaggle_Houseprices/blob/a1780c2a555de4925df51a4c3db96daeb3fae6a0/Model.ipynb
[git_knn]: https://github.com/jvanlier/Kaggle_Houseprices/blob/a1780c2a555de4925df51a4c3db96daeb3fae6a0/preprocess.py#L117
[git_autotrans]: https://github.com/jvanlier/Kaggle_Houseprices/blob/a1780c2a555de4925df51a4c3db96daeb3fae6a0/preprocess.py#L63
