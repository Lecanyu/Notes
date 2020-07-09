---
layout: post
title: Attention neural network
---

## ***Motivation***

When input data are massive and noisy, it may not be a good idea to directly train a model from the whole of original data. Because it is difficult for models to capture the meaningful information behind the massive data.

For example, in my previous work on jigsaw puzzle solving, it is important to transfer the calculation to the stiching area instead of the whole fragment. In NLP field, it is unnecessary to seize all context to translate a few of local words. 

Generally, human's perceptual system also focus on some particular areas to obtain information.


## ***Soft selection and hard selection***

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

LSTM are recurrent neural network modules, which convert the feature map into captions.

The intuition is that the attention model picks some input from feature map vector $$y_1, y_2, ..., y_n$$ (because softmax is easily dominated by the maximum one). 

If you have difficult to understand, go to the original introduction [HERE](https://blog.heuritech.com/2016/01/20/attention-mechanism/).


## ***Some insights about Structured Attention Networks*** 

Here I'd like to tell some insights about the paper "Structured Attention Networks". {% sidenote 2 'Kim, Yoon, et al. [Structured attention networks](https://arxiv.org/pdf/1702.00887.pdf). ICLR 2017'%}

The key contribution in this paper is that the authors use a CRF to model the attention layer.

In this paper, the author use below formulation to generalize the attention framework.

{% math %}
c = \mathbb{E}_{z \sim p(z|x,q)}[f(x,z)] = \sum_{i=1}^n p(z=i|x, q) x_i 
\quad \textsf{(original version in paper)} \\

z = \mathbb{E}_{s \sim p(s|y, c)}[f(y, s)] = \sum_{i=1}^n p(s=i|y, c)y_i
\quad \textsf{(use annotations in above pictures)} \\

{% endmath %}

For consistency, I will use the same annotation in above pictures to explain.
The $$s\sim p(s|y,c)$$ is the attention distribution. It assigns different weights to the input $$y_i$$. $$c$$ is the so-called query, which is the output $$h_1, h_2, ..., h_{k-1}$$ in above network structure (i.e. the medium output of RNN). $$f(y, s)$$ is annotation function which generate a output by combining original input $$y$$ and attention distribution $$s$$. In above example, the $$f(y, s) = ys$$.

In this paper, the authors proposed that we can apply a CRF to describe the relationship among all of $$y, s$$. As the figure showing below, the red box can be substituded by a CRF. Therefore, we will have 
{% math %}
z = \mathbb{E}_{s \sim p(s|y, c)}[f(y, s)] = \sum_C \mathbb{E}_{s \sim p(s_C|y, c)}[f_C(y, s_C)]
{% endmath %}
where the $$C$$ indicates the maximum clique.
The above example can be seen as a special case of this model, since the CRF allows the dependence between different $$s_i$$. Hence, it is more robust to describe the real probabilistic distribution.



{% maincolumn 'assets/machine_learning/structured_attention_network.png'%}



***Note: now my understanding may be wrong. I need to further read and double check.***



## Self-attention 

Self-attention concept may be introduced by paper [Attention is all you need](https://arxiv.org/abs/1706.03762). 
It is used in NLP task. 
There is a good and clear introduction [post](https://jalammar.github.io/illustrated-transformer/) about how self-attention works in Transformer. 
{% sidenote 1, "In NLP, the words are represented by embedded vectors via word2vec technique. There is a [post](https://medium.com/deeper-learning/glossary-of-deep-learning-word-embedding-f90c3cec34ca) to introduce this technique. Basically, the similar words (depends on how to define the similarity, e.g. concurrent appeared words are similar) will have closer distance in embedded space."%}

The core idea behind self-attention is to build some connection between current words with its context or even the whole long sentence.
In Transformer paper, they design some trainable parameters matrix to convert orginal word embedding into key, value, query vectors. 
Then use those key/value/query vectors to build the in-between connections via vector-matrix calculation.
