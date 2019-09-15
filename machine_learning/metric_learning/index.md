---
layout: post
title: Metric learning
---

Here we introduce several metric learning methods.

## Siamese network and its loss function 
{% maincolumn 'assets/machine_learning/siamese_net.png'%}
Siamese network is used to learn similarity of two inputs.
It feeds two inputs to two networks (these two nets have same architecture and weights) and output two feature vectors for similarity measurement (e.g. cosine, l2-distance). Then the measurement will be calculated by contrastive loss.

Here is the contrastive loss
{% math %}
L = (1-Y)\frac{1}{2}D_w^2 + Y\frac{1}{2}[\max(0, m-D_w))]^2	
{% endmath %}
where $$D_w$$ is similarity measurement. $$m$$ is a margin hyperparameter. $$Y=0$$ means the inputs should be similar.
$$Y=1$$ means the inputs should different, and when $$D_w$$ is larger than $$m$$, then there is no loss penalty. 



## Triplet network and its loss function
The triplet network is very similar with siamese network. It just uses three inputs: anchor, positive and negative instances.
{% maincolumn 'assets/machine_learning/triplet_net.png'%}
The triplet loss is 
{% math %}
L = \max(D(anchor, positive) - D(anchor, negative) + margin, 0)
{% endmath %}
where $$D(,)$$ is the similarity measurement (distance function).
When $$D(anchor, positive) - D(anchor, negative) < -margin $$, there is no loss penalty.
