---
title:  "Deep Turnaround"
client: Amsterdam Airport Schiphol
tech: > 
  Python (pandas, numpy, keras, matplotlib, seaborn, click, OpenCV), TensorFlow, TensorFlow Object Detection API, TensorBoard,
  CNNs / ConvNets, Object Detection, YOLO, SSD, Faster R-CNN, ResNets, Video Activity Recognition, Inception-based architectures, multi-task learning, Locality Similarity Hashing, Kalman filter tracking, Airflow, MLflow, Linux
from: 2018-08-01 00:00:00 +0200 
to: 2020-04-01 00:00:00 +0200
---
In a previous project to predict departure delays, it became apparent that the airport does not know as much as it could when it comes to the turnaround process. 
We took a bold move and decided to acquire the data ourselves. This became known as the "Deep Turnaround" project: an initiative to detect events during the aircraft turnaround process using state of the art deep learning-based computer vision techniques.

I built the initial prototype and evangelized it internally. The prototype used object detection, Kalman filter tracking and Python-based event generation code. As the team grew, I transitioned into a hands-on Lead Data Scientist role and spearheaded the rewrite into a more maintainable end-to-end learning system using activity recognition convolutional neural nets that operate on sequences of images.

