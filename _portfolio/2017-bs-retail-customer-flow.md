---
title:  "Retail Customer Flow"
industry: Retail
tech: Python (pandas, matplotlib), Java
methods: Smoothing, Kalman filter
from:  2016-12-01 00:00:00 +0000
to: 2017-06-01 00:00:00 +0000
---

## Context
This project was performed as part of [BlueSense][bs] product development. We already had mature technology to reconstruction people's positions inside a store, but that yielded little insight into how people actually move around.

My goal was to completely reconstruct each customer's entire physical journey inside a store, but there are many technical challenges to overcome. For instance, routing people around walls rather than through them turns out to be quite difficult (especially if you don't want to manually maintain data about those walls). A somewhat more achievable or intermittent goal was to simply understand the direction people move towards, at each point in the store. That was the scope of this project. We call this the "flow" of customers.

## Role
I made a proof of concept in December 2016 and proceeded to supervise an intern from February 2017 until June 2017.

## Approach
We had the BlueSense system up and running at a large retailer. 60 Wi-Fi sensors, 10.000 square meters, 5000 - 12000 daily visitors. The measurements were collected 24/7, in real-time, and positions were reconstructed using trilateration.

My approach consisted of grouping the reconstructed positions by device id, sorting them in time and calculating angles from each position to the next known position. These angles were then binned and averaged.

An important lesson learned early on is that this doesn't work very well without smoothing: the measurements (and thus the reconstructed positions) are simply too erratic. So I proceeded to look into position smoothing: from simple exponential moving averages to Kalman filters. The Kalman filters I experimented with ranged from simple filters that simply predict the next position as a weighted average with the previous state, to more complex filters that have a physical model of how objects move through space, and as a side-effect also estimate velocity. It turned out that a simple exponential moving average worked best, and so I implemented that in Apache Storm (Java).

When the intern took over, he improved my naive flow map construction a lot. He thoroughly researched multiple methods of measurement and location smoothing, a Gaussian Process approach for measurement resampling, and devised a much more sophisticated way of constructing flows: instead of looking for a dominant vector, he builds up the entire distribution of directions for each point. And instead of considering just the movements inside the bin, he also uses movements from surrounding bins (discounted based on their distance to the bin center). This was all done very well.

Leiden University awarded him with a high-but-justified grade of 9.5/10 and we submitted a paper for the Data Mining and Knowledge Discovery journal by Springer. Fingers crossed!

## Business value
So far, this has been primarily an academic pursuit. I imagine that it could be used to optimize layouts of physical spaces by reducing chokepoints and collision points (i.e. people moving in opposite directions) or to understand changes in behaviour through time. Slow sales could possibly be explained by people simply not walking past the items, which could drive decisions to change the store layout.

[bs]: {{ site.baseurl }}{% link _portfolio/2017-bs-bluesense.md %}
