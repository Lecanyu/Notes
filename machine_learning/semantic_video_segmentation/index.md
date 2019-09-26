---
layout: post
title: Semantic Video Segmentation
---


## Semantic segmentation 
The classic deep learning model for semantic segmentation is FCN.... 
<span style="color:red"> add more introducation about FCN.</span>


## Semantic video segmentation
For videos, a naive method is to segment the image frame one by one. 
But this strategy has several drawbacks: 
1. Inefficiency.
2. It doesn't take advantage of the video spatio-temporal information. 

The video spatio-temporal information can be utilized for calculating more stable segmentation results.

There are some works like [Feature Space Optimization for Semantic Video Segmentation](http://abhijitkundu.info/Publications/VideoFSO_CVPR16.pdf) adopt CRF to impose extra constraints.


