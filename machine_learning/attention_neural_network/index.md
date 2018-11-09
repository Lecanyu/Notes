---
layout: post
title: Attention neural network
---

## Motivation

When input data are massive and noisy, it may not be a good idea to directly train a model from the whole of original data. Because it is difficult for models to capture the meaningful information behind the massive data.

For example, in my previous work on jigsaw puzzle solving, it is important to transfer the calculation to the stiching area instead of the whole fragment. In NLP field, it is unnecessary to seize all context to translate a few of local words. 

Generally, human's perceptual system also focus on some particular areas to obtain information.


## Soft selection and hard selection

Researchers have realized the importance of attention, and they have proposed two approaches to fulfill attention mechanism. 

1. Soft selection.

	The selection (attention transferring) layer is differentiable, and thus the whole networks can be trained end to end.

2. Hard selection

	The selection layer is not differentiable. A typical implementation of this layer is  reinforcement learning.


Here are the representive network structures for these two type selections 
{% sidenote 1 'A nice explaination about attention network can be found [HERE](https://blog.heuritech.com/2016/01/20/attention-mechanism/)'%}

{% maincolumn 'assets/machine_learning/attention_network1.png'%}
Left picture: soft selection, right picture: hard selection, the random choice can be learned by a reinforcement learning.


In image captioning, the complete network structure can be below picture.
{% maincolumn 'assets/machine_learning/attention_network_global.png'%}

The attention model (purple blocks) is the selection layer. $$h_1, h_2, ..., h_{k-1}$$ is the input $$c$$ in above two pictures.

The intuition is that the attention model picks some input from feature map vector $$y_1, y_2, ..., y_n$$ (because softmax is easily dominated by the maximum one). 

If you have difficult to understand, check [HERE](https://blog.heuritech.com/2016/01/20/attention-mechanism/).


## Some insights about Structured Attention Networks 

Here I'd like to tell some insights about the paper "Structured Attention Networks". {% sidenote 2 'Kim, Yoon, et al. [Structured attention networks](https://arxiv.org/pdf/1702.00887.pdf). ICLR 2017'%}

The key contribution in this paper is that the authors use a CRF to model the attention layer.


***Now my understanding may be wrong. I need to further read and double check.***



