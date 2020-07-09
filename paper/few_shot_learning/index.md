---
layout: post
title: Few-shot Learning
---

In [my recent work](https://arxiv.org/abs/2001.08366) for few-shot learning, I have read a lot of paper in this or related fields.
I wrote some short summaries for them.

This post will be constantly update...  


## 


## Active Sampling
Original paper ''Active Sampling for Open-Set Classification without Initial Annotation''. 
{% sidenote 1, 'Original paper see [here](https://www.aaai.org/ojs/index.php/AAAI/article/view/4353). '%}

Key features:

这篇文章是关于主动学习而非小样本学习。我非常简略的浏览了一下，它想要解决的问题主要是open-set图像分类的问题，并且在没有pretrained model的情况下，希望能主动的选择代表性的数据进行标注，然后进行学习。

他们的实验所用的数据集相对简单，都是那种低分辨率的灰度图，比如Fashion-MNIST。

但这篇文章还是有一些值得借鉴的方法：

1.在没有pretrained model的情况下，如何选取有代表性的数据？
文章作者借鉴了transductive experimental design (TED)的方法，将数据选择问题formulate成了一个解线性优化的问题。
该方法有比较好的通用性，
但值得注意的是，由于没有pretrained model，他们直接拿原始图片作为特征向量进行优化，这会使得构造的矩阵维度很大，对于高分辨率的RGB图片不一定适用。


2.这篇文章用了一种叫low rank representation的方法来提取feature，而非深度学习中的CNN。
这种方法的基本假设其实与深度学习类似，就是

Data from the same class should be distributed in the same low-dimensional subspace. 
While the dimension of the subspace corresponds to the rank of the representation matrix, 
LRR tries to find the lowest-rank representation that can represent the data examples with linear combinations of given dictionary.

关于这个low rank representation方法，有时间的话看看相关的
[paper1](http://www2.egr.uh.edu/~zhan2/ECE6111/class/Latent%20Low-Rank%20Representation%20for%20Subspace%20Segmentationpdf.pdf)，
[paper2](https://zhouchenlin.github.io/Publications/2010-ICML-LRR.pdf)。

