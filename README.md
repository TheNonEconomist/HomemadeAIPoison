# HomemadeAIPoison
It's home-made and hurts bad AI... 


## What for? 
Failed attempt at building home-made AI poison pill. But AI became a paintbrush for art.
You can find some works here - https://gallery.so/neco. 

In a way I'm treating the process of modeling AI itself as an art.

## What was Achieved?
1) Various color filters made by controling number of training epochs.
2) Color contrast effects created by swapping out different activation functions.
3) Pixelation control done by placing 2D convolutional layers at the right places.



#### Disclaimers
Realistically, Data poisoning is a nuanced technique that involves - 
1) Duplicating data to make a model think data distribution is different that what it really is
2) Perturbing data with random noise to effect what a model learns per data point
3) Injecting false labels
4) Swapping out real data with fake data during transmission
5) And more. 

To make a model learn a distribution over a random noise would only be effective if one were to attempt to learn about the pseudo-randomness that a particular machine that generated poisoned data might be using. 

