---
title: "Predicted Off-Block Time (Departure Delay Prediction)"
client: Amsterdam Airport Schiphol
tech: > 
  Python (pandas, numpy, scikit-learn, matplotlib), Jupyter, SQL, Hive, Spark, Databricks.
  random forest regressor, boosted trees (XGBoost), LIME and SHAP 
from:  2017-10-01 00:00:00 +0200
to: 2018-07-01 00:00:00 +0200
---
Schiphol has experienced tremendous growth during the last couple of years, and infrastructure has struggled to keep up. The inevitable result is increased delays. I set out to develop a model to predict delay and - in the process - try to get a better understanding of the factors that drive delays. 

To maximize predictive accuracy, I used a boosted tree regression model with a ton of feature engineering. The model improved existing estimates of departure time by 15% to 50%. Interpretability suffered, of course, but part of it could be recovered using LIME and SHAP. 
Unfortunately, it became apparent that the airport did not have enough data about the progress of the turnaround process to make a really great prediction. This led to the Deep Turnaround project (see above).
