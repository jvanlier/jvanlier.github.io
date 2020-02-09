---
title:  "Predicted Off Blocks Time"
client: Schiphol
tech: > 
  Python (pandas, numpy, scikit-learn, matplotlib), Jupyter, SQL, Hive, Spark, Databricks.
methods: random forest regressor, boosted trees (XGBoost), Lime and SHAP 
has2x: true
from:  2017-10-01 00:00:00 +0200
to: 2018-07-01 00:00:00 +0200
---

## Context
Schiphol airport has about 750 departures a day. There has been tremendous growth during the last couple of years, and infrastructure has struggled to keep up. This results in delays.

When an arrival flight comes to a full stop at an aircraft stand, the timestamp is registered as *Actual On Blocks Time*. When the aircraft leaves towards the runway to take off, the *Actual Off Blocks Time* timestamp is registered. In between those two timestamps, the aircraft is unloaded, fueled, cleaned, boarded, etc - this is called the *turnaround process*. We aim to predict the departure time before the aircraft leaves, hence *Predicted Off Blocks Time*. The goal is to help Operations gain a better understanding of departure delays, which should improve gate planning and increase operational efficiency. 

## Team
I was hired as a Senior Data Scientist in the Data Innovation Lab at Schiphol. The project team was supported by one to two Data Engineers and occasionally by a Junior Data Scientist.

## Approach
In some sense, this was a traditional regression problem: take a couple years of historic data, with historic features about flights such as airline, number of passengers, number of seats, size of aircraft, time of day, day of week, etc. Then simply predict a numeric quantity: the departure delay, and add that to the Scheduled Off Blocks Time.

The largest complicating factor was that data about flights changes all the time. The number of passengers frequently changes when the aircraft is already boarding. The gate may change just before arrival, occasionally adding a few minutes minutes taxi time, which could propagate into a departure delay. But most importantly: new data becomes available as the turnaround progresses. Data that was `NULL` initially, now suddenly has a lot of predictive power. For instance, a flight might arrive perfectly on time, with no problems whatsoever. Then, boarding starts 20 minutes late! Surely we now need to update our predictions to incorporate this new information. But of course, we're not allowed to cheat by already using this data when it was not available yet.

So, to properly expose the model of all the previous data of a flight, we needed to reconstruct the historic states for each flight based on a log of the changes. This was a humungous ETL effort, but we managed to pull it off.

The modelling itself then was quite simple: linear models at first, then XGBoost since there was value to exploit from feature interactions, then Random Forest in order to get some measure of uncertainty on the predictions (prediction intervals). A relatively large amount of time was spent on building tools to explain predictions, with Lime and SHAP. 

At this point, after first creating a Python Flask based PoC, the model was handed over to the engineering folks, who used Spark Streaming to productionize it.

## Business value
Improved existing estimates of departure time (Target Off Blocks Time, Estimated Off Blocks Time, Target Startup Approval Time) by 15% - 50%. The model was deployed to production, but business adoption remained a challenge.

An important lesson learned was that that the airport knows very little about the turnaround process itself. All that data remains with the airline and handlers. That keeps the turnaround process relatively unpredictable from the airport's perspective. We hypothesize that detailed data about the turnaround process can help diagnose inefficiencies, root causes, and create much better features for the Predicted Off Blocks Time model. This led to the Deep Turnaround project.
