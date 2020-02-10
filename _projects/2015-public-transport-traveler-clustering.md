---
title:  "Public Transport Traveler Clustering"
client: "Regional Dutch public transport provider (via KPMG)"
tech: > 
  Python (pandas, numpy, scikit-learn, matplotlib), Jupyter notebook, SQL, Hive, Hadoop, Spark, 
  Hortonworks big data cluster, Linux, Gephi,
  k-means clustering, dimensionality reduction (PCA)
from:  2015-07-01 00:00:00 +0200
to: 2015-09-15 00:00:00 +0200
---
Since switching to electronic payment cards for public transport, a lot of data has been collected on behavior of travelers. 
This raises the question: can this data be utilized to create better products, more in line with travelers' wishes?

I investigated how (anonymized) travelers can be assigned into several clusters based on their behavior. I used Hive on a Hadoop cluster to calculate various 
normalized behaviour indicators, applied the K-means clustering algorithm, visualized the results with matplotlib and Gephi and 
drove the interpretation and validation of the results with business stakeholders.
