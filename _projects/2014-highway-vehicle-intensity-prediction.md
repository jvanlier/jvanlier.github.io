---
title:  "Highway Vehicle Intensity Prediction"
client: "Ministry of Infrastructure and Environment (<i>Rijkswaterstaat</i>; via KPMG)"
tech: > 
    Python (pandas, numpy, matplotlib, scikit-learn), Jupyter notebook,
    random forest regressor, gradient descent, time series prediction, autoregressive feature extraction
from:  2014-06-01 00:00:00 +0200
to: 2014-08-01 00:00:00 +0200
---
The Dutch road administration has many terabytes of data from measurements of vehicles on the highways, made using induction loops embedded in the road. This results in noisy measurements of the number of vehicles, their length and velocity. The goal of this project was to investigate the possibilities of applying big data techniques to induction loop sensor data. 

I developed a predictive model for the intensity on the road at any given time, based on historic intensity and weather data. The model was able to predict the standard weekly pattern quite accurately, including holiday effects and rush hour traffic. 
Adding precipitation data reduced the error by 3%. Predicting traffic jams due to collisions and rare "black swan" events remains elusive, though. 
 This was just a Proof-of-Concept, but the results were featured in a 
[newspaper article in the NRC](https://www.nrc.nl/nieuws/2014/11/14/als-je-files-kunt-voorspellen-kun-je-ze-ook-sture-1437964-a1279877)  (Dutch).
