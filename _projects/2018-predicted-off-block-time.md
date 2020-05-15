---
title: "Predicted Off-Block Time (Departure Delay Prediction)"
client: Schiphol Group
tech: > 
  Python (pandas, numpy, scikit-learn, matplotlib, Flask), Jupyter, SQL, Hive, Spark, Spark Structured Streaming, Databricks,
  random forest regressor, boosted trees (XGBoost), LIME and SHAP 
from:  2017-10-01 00:00:00 +0200
to: 2018-07-01 00:00:00 +0200
---
Schiphol has experienced tremendous growth during the last couple of years, and infrastructure has struggled to keep up. The inevitable result is increased delays. I set out to develop a model to predict delay and - in the process - try to get a better understanding of the factors that drive delays. 

The final model that maximized predictive accuracy was a boosted tree (XGBoost) model with a lot of feature engineering. The model improved existing estimates of departure time by 15% to 50%. I built an async flight API client, which refreshes on a timer and shows predictions in simple Flask UI. Later on, I built the first version of a low-latency streaming implementation using Spark Structured Streaming on Databricks. 

