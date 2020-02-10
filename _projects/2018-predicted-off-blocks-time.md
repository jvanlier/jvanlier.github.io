---
title:  "Departure Delay Prediction (Predicted Off Blocks Time)"
client: Schiphol
tech: > 
  Python (pandas, numpy, scikit-learn, matplotlib), Jupyter, SQL, Hive, Spark, Databricks.
  random forest regressor, boosted trees (XGBoost), Lime and SHAP 
from:  2017-10-01 00:00:00 +0200
to: 2018-07-01 00:00:00 +0200
---
Schiphol airport has about 750 departures a day. There has been tremendous growth during the last couple of years, and infrastructure has struggled to keep up. The invitable result is increased delays. I was hired as a Senior Data Scientist in the Data Innovation Lab at Schiphol and tasked to build this model, with the support of  one to two Data Engineers and a Junior Data Scientist. 

The resulting model was a boosted tree regression model (XGBoost) with a ton of feature engineering to extract maximum value out of historic data. The model improved existing estimates of departure time by 15% to 50% and featured explanations through SHAP to facilitate business acceptance. Unfortunately, the airport did not have enough data about the progress of the turnaround process to make a really great prediction. This led to the Deep Turnaround project.
