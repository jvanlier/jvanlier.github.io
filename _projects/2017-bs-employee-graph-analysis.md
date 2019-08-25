---
title:  "Employee Graph Analysis"
industry: Offices, Facility Management
tech: Spark, Hive, Python, Gephi
methods: Graph construction, Jaccard index, ForceAtlas2, Modularity (Network Science)
from:  2017-06-30 00:00:00 +0200
to: 2017-06-01 00:00:00 +0200
---

## Context
This project was performed as part of [BlueSense][bs] product development. When BlueSense moved into the office space sector, we were looking for creative ways to add value with the data at hand. This was a proof-of-concept at the time, with the purpose of giving a sneak peak into future potential of BlueSense.

## Team
Sole Data Scientist.

## Approach
I gathered 3 weeks of location data, consisting of nothing more than anonymized device identifiers, timestamps and locations in 2 dimensions (x, y). I used that to construct a co-occurrence matrix: a square table in which each device is represented by a 1 row and 1 column. The values within the matrix range from 0 to 1, where a 0 indicates that the two devices have never been seen in close proximity of each other and a 1 indicates that they are inseparable. I used a metric inspired by the Jaccard index to calculate these quantities.

When representing this matrix as a undirected graph in which each device is a node, with edge weights representing the co-occurrence index, an interesting structure is revealed by using the ForceAtlas algorithm. It finds an equilibrium of node locations that naturally brings out the company's team structure, since people in the same team are often seen together.

The teams are colored using a network community detection technique based on the modularity metric.

## Business value
- Identify devices that belong to the same person (to avoid double counting).
- Identify teams & team sizes. This can be used to find an optimal team-to-workspace allocation.
- Quantity collaboration between teams. Want IT and business to "align" more often? Use this method to measure whether (physical) collaboration actually improves over time.

[bs]: {{ site.baseurl }}{% link _projects/2017-bs-bluesense.md %}
