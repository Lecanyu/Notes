---
layout: post
title: Human Pose Estimation
---

There are some popular paper about human pose estimation.
{% sidenote 1, 'There is a nice human pose estimation overview in [here](https://zhuanlan.zhihu.com/p/85506259). '%}
I briefly introduce them here so that I can review quickly in the future.

This is not an exhausted list and I will update it constantly.


## Concept overview
Single human pose estimation: estimate body keypoints in a single person.

Multiple human pose estimation: estimate body keypoints in mutiple persons.

Single human pose estimation can be seen a special case of multiple estimations. 
Current research interest is in deep learning based multiple human pose estimation (previously, there are some traditional methods which is based on hand-crafted features and graphical models).
To solve it, there are two ways:

1.top-down menthod which detects and crops single person one by one and then applies single human pose estimation method.

2.Button-up method which detects all possible keypoints regardless which person they belong and then solve the association problem (i.e. associate those keypoints with person instance).

Since Top-down methods take advantage of object detection results as the prior, it is usually more accurate but kind of slow.
Whereas buttom-up methods are more unified (without person detection) and fast with the cost of accuracy. 

Something more:

In deep learning based human pose estimation, researchers previously focused on image context, but now they gradually realize that the resolution of image and feature map is crucial in detection performance. 
Because the keypoint detection is a pixel-level detection which is fine-grained. 
To solve it, a trivial approach is to directly resize original image to a larger resolution.
And there is a recent surge of interest in network design for high resolution feature representation.
The [paper](https://arxiv.org/pdf/1908.07919.pdf) (Deep high-resolution representation learning for visual recognition) discusses this aspect.



## CPM

Original paper ''Convolutional Pose Machines''. 
{% sidenote 1, 'Original CPM paper see [here](https://arxiv.org/pdf/1602.00134.pdf). '%}

Key features:

1.CPM is a single human pose estimation method.

2.It designs a sequential composition of convolutional architectures which model the spatial relationship via large receptive field. 
Besides, it enforces intermediate supervision (i.e. loss optimization) to relieve gradient vanish problems.

{% maincolumn 'assets/paper/human_pose_estimation/CPM1.png'%}

3.Instead of regressing keypoint locations, it introduces heatmap as the training target.
Heatmap has several advantages:
(1) it introduces spatial uncertainty.
(2) Unlike 1d location in regression, heatmap is 2d representation which implicitly encodes spatial information.
As a result, the heatmap can be fed into following convolutional stages as an input.

Note that training dataset only provide the keypoint location groundtruth, which can be trained as target.
The authors apply Gauss kernal on groundtruth points to generate the groundtruth heatmap.



## Hourglass 

Original paper ''Stacked Hourglass Networks for Human Pose Estimation''. 
{% sidenote 1, 'Original Hourglass paper see [here](https://arxiv.org/pdf/1603.06937.pdf). '%}

Key features:

1.It is a single person estimation method.

2.This paper design a novel network architecture, called stacked hourglass. 
The authors claim this architecture can capture local and global features better.




## Associative Embedding 

Original paper ''Associative Embedding: End-to-End Learning for Joint Detection and Grouping''. 
{% sidenote 1, 'Original Associative Embedding paper see [here](https://arxiv.org/pdf/1611.05424.pdf). '%}

Key features:

1.It is a multiple person estimation method. 

2.It estimates all limb keypoints and associative embedding simultaneously. 
Then, it uses the associative embedding values to group those keypoints. 

{% maincolumn 'assets/paper/human_pose_estimation/associative_embedding1.png'%}


3.It uses the stacked hourglass as the network arthitecture and designs new associative embedding loss which enforces the keypoints in the same person have similar embedding values.
The loss is described as below:

{% math %}
\bar h_n = \frac{1}{K} \sum_k h_k(x_{nk})
{% endmath %}
where $$h_k \in R^{W\times H} $$ is the predicted embedding value heatmap for $$k$$-th body joint (total $$K$$ body joints).
$$h(x)$$ is the embedding value at location $$x$$.
$$x_{nk}$$ is the groundtruth location of the $$n$$-th person's $$k$$-th body joint.

So the equation above calculates the mean embedding value of all $$K$$ body joints at groundtruth location for $$n$$-th person (total $$N$$ person in an image).

The embedding associative loss is defined as 
{% math %}
L_g(h, T) = \frac{1}{N} \sum_n \sum_k (\bar h_n - h_k(x_{nk}))^2 + \frac{1}{N^2} \sum_n \sum_{n^{'}} \exp\{-\frac{ (\bar h_n - \bar h_{n^{'}})^2 }{2\sigma^2}\}
{% endmath %}
The first term means that all body joints in one person should be closed to $$\bar h_n$$.
The second term means that the mean embedding value of different person should be different (large difference gives small loss).



## OpenPose 

Original paper ''OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields''. 
{% sidenote 1, 'Original OpenPose paper see [here](https://arxiv.org/pdf/1812.08008.pdf). '%}

I just quickly browse this paper without careful reading. 
If need, I'll read this paper again. 

Key features:

1.This paper is a buttom-up method: using CPM (convolution pose machine architecture) as the backbone to predict keypoint heatmap and keypoint association simultaneously.

2.It proposes part affinity field (PAF) to represent keypoint association.
PAF is a vector field which indicates the keypoint connected direction. 
The vector (i.e. connected direction) will be treated as score in solving bipartite matching.

{% maincolumn 'assets/paper/human_pose_estimation/openpose1.png'%}




## HRNet 

Original paper ''Deep High-Resolution Representation Learning for Human Pose Estimation''. 
{% sidenote 1, 'Original HRNet paper see [here](https://arxiv.org/pdf/1902.09212.pdf). '%}

Key features:

1.Researchers realize that the image or feature map resolution is crucial in keypoint detection performance.
So the author design a new network architecture for obtaining high resolution feature representation.

{% maincolumn 'assets/paper/human_pose_estimation/HRNet1.png'%}


2.In original paper, this method is a top-down method, but it should be easily modified for buttom-up estimation like associative embedding.

3.This paper still enforces intermediate loss optimization. 




## HHRNet 

Original paper ''HigherHRNet: Scale-Aware Representation Learning for Bottom-Up Human Pose Estimation''. 
{% sidenote 1, 'Original HHRNet paper see [here](https://arxiv.org/pdf/1908.10357.pdf). '%}

Key features:

1.This paper is mainly based on HRNet. It propose a higher resolution of feature representation via deconvolution operation.

{% maincolumn 'assets/paper/human_pose_estimation/HHRNet1.png'%}
The red box in figure is their contribution.

2.This method demonstrates the buttom-up approach with higher resolution of feature representation.
It adopts associative embedding strategy for keypoints grouping.




