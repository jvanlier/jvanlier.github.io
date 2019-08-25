---
title:  "BlueSense"
industry: Facility Management, Retail, Public Transport, Events (soccer, music, trade fairs)
tech: Python, Java, MongoDB, Kafka, Spark, Hadoop, HDFS, Hive, Storm
from:  2014-01-02 00:00:00 +0000
to: 2017-06-10 00:00:00 +0000
---
BlueSense measures the behaviour of people within buildings, in real-time, using mainly Wi-Fi sensors. It includes state of the art analytics to understand crowd density, crowd movement patterns, dwell times and occupancy levels, both descriptive and predictive. This system was deployed at various clients in retail, public transport, facility management and a football stadium.

This was my main project during my time in the Big Data & Analytics team at KPMG. I started out doing mostly engineering and gradually transitioned into leading the analytics effort as of October 2016. However, in a small start-up like team, scrambling to build a product and find product-market fit before cash runs out, it's impossible to have neatly separated responsiblities like in a large corporate. This has given me broad experience in: 

- dealing with sensor hardware 
- creating sensor data acquisition software
- ingesting said data in real-time on a cluster
- collaborating with a Trusted Third Party to create a privacy-by-design system with pseudonymization and an opt-out facility
- designing, installing and maintaining a big data cluster based on the lambda architecture
- building production-grade analysis code
- prototyping a frontend dashboard and subsequently managing an external frontend developer
- prototyping an indoor navigation app 
- building a visitor prediction app for the KPMG restaurant
- reaching out to customers, understanding their requirements, validating our results with them
- being part of the management team, making commercial and HR decisions

It was an amazing and sometimes nerve-wrecking experience. BlueSense has been incubating within KPMG NL from early 2014 until June 2017. This eventually culminated into a spin-off, which unfortunately did not gain enough traction.

Since this project pivoted multiple times, featured subprojects are handled separately:

- [Retail Customer Flow][flow]
- [Employee Graph Analysis][graph]
- Workplace Occupancy Monitoring - coming soon!
- Retail Customer Analytics - coming soon!
- Real-Time Crowd Prediction - coming soon!
- Indoor Navigation App - coming soon!

[flow]: {{ site.baseurl }}{% link _projects/2017-bs-retail-customer-flow.md %}
[graph]: {{ site.baseurl }}{% link _projects/2017-bs-employee-graph-analysis.md %}
