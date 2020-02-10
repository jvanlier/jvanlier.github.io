---
title:  "Departure Delay Prediction (Predicted Off Blocks Time)"
client: Amsterdam Airport Schiphol
tech: > 
  Python (pandas, numpy, scikit-learn, matplotlib), Jupyter, SQL, Hive, Spark, Databricks.
  random forest regressor, boosted trees (XGBoost), LIME and SHAP 
from:  2017-10-01 00:00:00 +0200
to: 2018-07-01 00:00:00 +0200
---
Schiphol has experienced tremendous growth during the last couple of years, and infrastructure has struggled to keep up. The invitable result is increased delays. I set out to develop a model to predict delay and - in the process - try to get a better understanding of the factors that drive delays. 

I settled on boosted tree regression model (XGBoost) with a ton of feature engineering to extract maximum value out of historic data. The model improved existing estimates of departure time by 15% to 50% and supported explanations through LIME and SHAP to facilitate business acceptance. During the process, I learned that the airport does not have enough data about the progress of the turnaround process to make a really great prediction. This led to the Deep Turnaround project (see above).
