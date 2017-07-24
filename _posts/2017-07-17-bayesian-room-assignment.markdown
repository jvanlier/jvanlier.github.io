---
layout: post
title:  "Bayesian Room Assignment"
date:   2017-07-17 12:00:00 +0200
categories: blog
math: true
---
This first post is about using Bayes' theorem to assign devices to rooms probabilistically, based on noisy measurements. I like it because it's a very practical application of Bayes' rule. Examples like these always resonate more with me than the artificial examples on Wikipedia about things like drug use.

Here's the problem. In the meeting room module in [BlueSense][bs], we're collecting data from a Wi-Fi sensor system that measures the presence of Wi-Fi client devices. The systems measures the MAC address, which uniquely identifies a Wi-Fi device, and the strength of the received Wi-Fi signal. There is a sensor in every room, but there aren't enough sensors to apply an advanced reconstruction technique such as trilateration.

A naive approach to determine in which room a device is, involves ranking the measurements on signal strength, taking the highest signal, finding out in which room the corresponding sensor is, and assigning the device to that room. Simple, right? This actually works ok-ish, because the walls between the rooms dampen the signal a bit, leading the room that the device is in to usually come out on top. But every now and then the system measures a signal that's way off, which causes us to assign the device to the wrong room. We need some way to deal with the noise. Bayes' theorem is well suited to these kinds of problems, as we shall see. 

## Bayes' rule 
[Bayes' rule][bayes] is stated mathematically as follows:

$$P(A \vert B) = \frac{P(B \vert A) P(A)}{P(B)}$$

Let's assume for a minute that there are only two rooms named A and B, and a "no room" option named N. Let's create the equations for calculating the probability that the device is in A. We get: 

$$P(i_A \vert m_A) = \frac{P(m_A \vert i_A) P(i_A)} { P(m_A) }$$

where $$i_A$$ should be read as "is in A" and $$m_A$$ as "measured in A".
This makes $$P(i_A)$$ the probability that a device is in A and $$P(m_A)$$ the probability that a device gets measured in A.

The conditional probability $$P(m_A \vert i_A)$$ describes the probability that a device gets measured in A if it is indeed in A, and $$P(i_A \vert m_A)$$ describes the probability that we're interested in: whether the device is actually in A, given that it was measured in A. This is called the posterior probability.

The denominator $$P(m_A)$$ can be expanded to:

$$P(i_A \vert m_A) = \frac{P(m_A \vert i_A) P(i_A)} { P(m_A  \vert  i_A) P(i_A) + P(m_A  \vert  i_B) P(i_B) + P(m_A  \vert  i_N) P(i_N) }$$

Two new probabilities popped up in the denominator. $$P(m_A \vert i_B)$$ is the probability that we see a measurement in A, given that the device is actually in B. Likewise, $$P(m_A  \vert  i_N) P(i_N)$$ is the probability that we see a measurement in N (no room at all) even though the device is in B.

What if we get a measurement in room B? The posterior that the device is actually in A is given by: 

$$P(i_A \vert m_B) = \frac{P(m_B \vert i_A) P(i_A)} { P(m_B  \vert  i_A) P(i_A) + P(m_B  \vert  i_B) P(i_B) + P(m_B  \vert  i_N) P(i_N) }$$

Only the numerator changes on the right side. The denominator is basically used to normalize the output between 0 and 1.

The equation for the event that a measurement is seen in N can be constructed in the same way.

That's it for room A. The probabilities for B and N are calculated in the same way, just make the appropriate substitutions for $$i_A$$ on the left side of the equation and in the numerator on the right side. 

### Priors
In order to evaluate these equations, we need to quantify our prior beliefs for $$P(i_A)$$, $$P(i_B)$$ and $$P(i_N)$$. Let's assume for now that it's equally likely for a device to be in any of the three locations, so $$\frac{1}{n}$$ where $$n$$ is the amount of rooms + 1 (to account for the "no room" case). So, $$\frac{1}{3}$$ for all three priors. 

This initial prior value is only used the very first time we see a device. For subsequent measurements, we use the posterior calculated previously as the new prior.

Finally, we need to think about the probability that a device gets measured in room A, if it is indeed in room A: $$ P(m_A \vert i_A) $$. 
This is obviously based on multiple factors such as room size, sensor placement, etc. Empirically, we have found this to be approximately equal to $$\frac{3}{5}$$.
 
For simplicity, we'll assign $$\frac{1}{n-1} \times (1-\frac{3}{5})$$ to the probabilities that we get measurements in B or N while the device is in A. That would be $$\frac{1}{5}$$ in this example with $$n=3$$. To improve this further, we could make this inversely proportional to the distance from room A to the other locations (i.e. the probability decreases if the room is further away), but I like to keep things simple as long as possible. Additional complexity can always be introduced later later if our simple assumptions don't seem to work.

## A Python-based simulation

### The setup
We have 1 sensor in the middle of each room and a bunch in the hallway (N). There are 7 rooms in total, so $$n = 8$$. 

![Setup](/assets/img/bayes_rooms/setup.png)

Let's start by simulating measurements for a device that's actually in room B. After each measurement, the probabilities for each room are computed. Remember that measurements are noisy, so even though the device is in B, occasionally we'll get something in A or C. After 10 measurements, the device moves across the hall to room E and we simulate 10 more. 

### Code for Bayesian updates

{% highlight python %}
import pandas as pd

# I have omitted the polygon definition in the region DataFrame,
# because the plotting code is part of a proprietary library.
regions = pd.DataFrame({'id': ['A', 'B', 'C', 'D', 'E', 'F', 'N']}) \
              .set_index('id')

priors = {region_id: 1/len(regions.index) for region_id in regions.index}
prob_measured_in_actual_room = 0.60
prob_measured_in_diff_room = (1 - prob_measured_in_actual_room) / \
                             (len(regions.index) - 1)
min_prob = 0.001

# (actual, measured)
sim_pairs = [('B', 'B'), ('B', 'A'), ('B', 'B'), ('B', 'C'), ('B', 'C'), 
             ('B', 'B'), ('B', 'B'), ('B', 'B'), ('B', 'B'), ('B', 'B'),
             # Moving across the hall
             ('E', 'E'), ('E', 'E'), ('E', 'F'), ('E', 'E'), ('E', 'D'), 
             ('E', 'F'), ('E', 'E'), ('E', 'E'), ('E', 'D'), ('E', 'E')
            ]

def calc_posterior(priors, measured_region_id):
    posteriors = {}
    
    for region_id in regions.index:
        if region_id == measured_region_id:
            posteriors[region_id] = \
                prob_measured_in_actual_room * priors[region_id]
        else:
            posteriors[region_id] = \
                prob_measured_in_diff_room * priors[region_id]

    total = sum(posteriors.values())
    
    # Normalize:
    for region_id, value in posteriors.items():
        posteriors[region_id] = value / total

        # Keep the posterior from tending towards zero:
        if posteriors[region_id] < min_prob:
            posteriors[region_id] = min_prob
    
    return posteriors

def simulate(priors):
    probabilities = priors

    for i, (actual_region_id, measured_region_id) in enumerate(sim_pairs):
        probabilities = calc_posterior(probabilities, measured_region_id)

        print_probabilities(i, probabilities)
        
        # Cannot include the plotting code unfortunately because it's in
        # a propietary library.
        # plot_map_with_probabilities(i, probabilities)
        
def print_probabilities(i, probabilities):
    print("Iter {}: {}".format(i+1, 
        {k: round(v, 2) for k, v in probabilities.items()}))
        
simulate(priors)
{% endhighlight %}

#### Dealing with posteriors that tend to zero
As we gather more and more evidence in favor of a specific room, at some point the probabilities for the rooms that are far away will tend to 0. It will take a very long time or be practically impossible for the model to 'drop' the belief that a device is in a specific room. 
This is not the desired behaviour; after all we're still in a dynamic situation in which a device *can* actually move to a different room at some point in time.
To compensate for this, we've put a minimum bound on probabilities of 0.01. It's not entirely pure from a statistical point of view, but it works well in this case.


### Result

![Result](/assets/img/bayes_rooms/result.gif)

The green star shows the actual device location, the purple room shows where the measurement was and the text inside the room indicates the room name and the probability that the device is in that room.

## Thoughts
Alright, this shows the desired behavior: we don't immediately assign a device to the latest known room if there's a strong prior belief that this device is somewhere else. But we can still convince it that it did in fact move, after a couple of measurements have been received. Much better than the naive solution we've been using so far. 


{::comment}
## Other approaches
There's various other ways to compensate for erratic sensor measurements, such as Kalman filters or plain old moving averages. I haven't done that yet, but if I did, I'd still want to the probabilistic region assignment - it quite nicely quantifies a belief about a device being in a certain room, while incorporating what is known about that device. The reduced variance on the measurements should be reflected in the priors that get picked.
{:/comment}




[bs]:           http://bluesense.io
[bayes]:        http://en.wikipedia.org/wiki/Bayes%27_theorem 
