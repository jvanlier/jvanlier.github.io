---
title:  "Highway Vehicle Intensity Prediction"
client: "Rijkswaterstaat / Ministry of Infrastructure and Environment (via KPMG)"
industry: Government, Ministry of Infrastructure and Environment
tech: Python (pandas, numpy, matplotlib, scikit-learn), Jupyter notebook
methods:  Random forest regression, gradient descent, time series prediction, autoregressive feature extraction
from:  2014-06-01 00:00:00 +0200
to: 2014-08-01 00:00:00 +0200
---
A colleague and I were given a dataset from the Dutch road administration, which was measured using induction loops in 
the road. This results in noisy measurements on the amount of vehicles, their length and their velocity. We combined
this with weather data from the Dutch weather service (KNMI) (both open data and high resolution weather radar dataset).

The goal of this project was to investigate the possibilities of applying big data techniques to induction loop 
sensor data. 

My colleague investigated the effect of weather conditions on traffic intensity (the amount of vehicles on the road per 
hour) and velocity.

I developed a predictive model for the intensity on the road at any given time, based on historic intensity data and 
weather data. This was a time series prediction problem that incorporated external data sources. I was able to predict 
the standard weekly pattern quite accurately, including holiday effects and rush hour traffic, for each day. 
Adding precipitation data improved the accuracy with 3%. Predicting traffic jams due to collisions and rare "black 
swan" events remains elusive, though. This system could also be used as a real-time anomaly detection system.

This was just a Proof-of-Concept, but the results were published in a 
[newspaper article in the NRC](https://www.nrc.nl/nieuws/2014/11/14/als-je-files-kunt-voorspellen-kun-je-ze-ook-sture-1437964-a1279877)  (Dutch).
