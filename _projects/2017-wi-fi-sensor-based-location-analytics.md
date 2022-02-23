---
title:  "Wi-Fi-Sensor Based Location Analytics"
tech: >
    Python (numpy, pandas, matplotlib, scikit-learn, flask, jupyter), Java, JavaScript, Hadoop, Kafka, Storm, Spark, HDFS, Hive, MongoDB, Prometheus, Git, Jenkins, Linux, AWS, GCP, MCMC (PyMC3), time series forecasting (Machine Learning).
client: KPMG
from:  2013-12-01 00:00:00 +0000
to: 2017-09-01 00:00:00 +0000
---
Part of a team that developed a system to measure the presence of Wi-Fi radios using custom Wi-Fi sensors. In essence, this enables insight into the approximate number of people in an area, how long they remain there, which route they take, how often they come back, and so on. This system was deployed at various clients in retail, public transport, facility management and a football stadium.

I started out on the Data Engineering side and gradually transitioned into Data Science. 

My engineering contributions:
- Designed, implemented and maintained a lambda-architecture big data platform. 
- Developed streaming data processing code and real-time summary statistics using Apache Storm (Java).
- Built a framework to greatly simplify PySpark-based batch jobs. Also, scheduling and monitoring these jobs.
- Privacy by design: developed the anonymization pipeline involving a Trusted Third Party and a physical opt-out facility.
- Built a Python/Flask/MongoDB/jQuery/Bootstrap based configuration management tool to simplify the administration of sensor locations, regions, maps, geometry, etc. This saved many hours of menial work.
- Monitoring with Prometheus.
- Supervising Junior Data Engineers. 

Science:
- Developed a real-time Crowd Monitor for the KPMG Restaurant, with a short-term prediction (30 mins ahead). This was beneficial for internal marketing and helped our colleagues avoid crowds and queues. 
- Analyzing data from 120 sensors in a large furniture store, working together with stakeholders to extract useful insights in shopper's behavior.
- Pivoting the product into a version for workplace utilization and occupancy monitoring, in order to more efficiently allocate teams to areas, and to possibly close down an entire section of the building (saving a lot of money on exploitation costs).

Fun side project:
- Prototyped an indoor navigation app for Android using iBeacons. Dijkstra-based routing, sensor fusion, proximity-triggered messaging managed by a Python backend.

