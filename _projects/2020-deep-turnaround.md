---
title:  "Deep Turnaround"
client: Schiphol Group
tech: > 
  Python (pandas, numpy, keras, matplotlib, seaborn, click, OpenCV), TensorFlow, TensorFlow Object Detection API, TensorBoard,
  CNNs / ConvNets, Object Detection, YOLO, SSD, Faster R-CNN, ResNets, video activity recognition, Inception-based architectures, multi-task learning, Locality Similarity Hashing, Kalman filter tracking, Airflow, MLflow, Spark, Databricks, PostgreSQL, Linux, Azure
from: 2018-08-01 00:00:00 +0200 
to: 2020-03-31 00:00:00 +0200
---
Deep Turnaround was an initiative to detect events during the aircraft handling process with Deep Learning-based Computer Vision. I built the initial prototype and evangelized it internally. When we got funding to proceed, my main responsibility was detection with high accuracy: initially by implementing Object Detection models using TensorFlow Object Detection API, Kalman-filter based tracking and rule-based event detection. Later on, this transitioned into end-to-end learning on small video clips, using a custom Action Recognition approach based on a DeepMind paper. 

Next to that, I also took on a significant chunk of engineering: ETL pipelines (from videos to de-duplicated pre-processed images), low-level TensorFlow code, Airflow DAGs for hyperparameter searches and experimentation, annotation tooling and maintaining Linux systems (on-prem and Azure VMs).

In addition, I played an important role in getting the project off the ground: convince legal to get permission to use the data, convince other stakeholders, determine and validate anonymization strategies, figure out optimal camera positions, perform site surveys, build a strong team, define the roadmap and team priorities.

Last I checked early 2022, the project is still going strong and has survived the COVID pandemic!

