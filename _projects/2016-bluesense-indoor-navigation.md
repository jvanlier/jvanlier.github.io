---
title:  "BlueSense: Indoor Navigation"
client: "KPMG"
industry: Developed as an in-house demo, but could be applied in airports, hospitals, big retail stores, malls, etc.
tech: Java, JUnit, SQLite, Android SDK, iBeacon, Python, Django
methods: High volume Beacon data aggregation, Dijkstra's algorithm
from:  2015-12-01 00:00:00 +0200
to: 2015-12-31 00:00:00 +0200
---
Done as part of BlueSense.

While it's not my ambition to be a Java/Android developer, I do like a software engineering challenge every now and 
then. A colleague and I set out to build a Proof-of-Concept Android app for indoor navigation, using a combination 
of signals from Bluetooth Low Energy beacons and the device's accelerometer data.

We had 20 beacons that were powered via the grid instead of via batteries. This allowed us to set them to a very high
frequency of 20 Hz (so 50 signals per sec per beacon), which actually made this work quite well (as long as there 
weren't too many obstacles or other people around).
 
The app uses trilateration to determine the user’s location, which is then displayed on a map. The app also triggers 
a configurable message when the user gets close to a beacon. 

I developed the data acquisition mechanism, the database 
connectivity, the graphical interface, an implementation of Dijkstra’s algorithm to find the shortest path to a 
destination, the proximity-triggered messaging and a Python/Django based backend. This app was developed in a small 
Scrum team with 2 developers.
