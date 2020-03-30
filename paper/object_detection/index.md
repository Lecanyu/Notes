---
layout: post
title: Object Detection
---

There are some popular paper about object detection.
I briefly introduce them here so that I can review quickly in the future.


## ATSS
Original paper ''Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection''. 
{% sidenote 1, 'Original ATSS paper see [here](https://arxiv.org/pdf/1912.02424.pdf). '%}

Key features:

1.One-stage (i.e. proposal-free) anchor based detector.

2.This paper conduct a lot of ablation experiments to study why there is gap between anchor-based and anchor-free detectors.
It uses RetinaNet and FCOS as the example to conduct experiments. 
By those experiments, the authors found that tiling multiple different size anchors in each location is unnecessary.
And the authors think that how to define positive and negative training samples is the essential difference between anchor-based and anchor-free detectors.

{% maincolumn 'assets/paper/object_detection/ATSS1.png'%}


3.This paper also gives a nice summary for related object detection works.


4.Based on IOU statistic, the author proposes an adaptive training sample selection strategy which automatically decide what samples should be positive and negative with almost hyperparameter-free.

{% maincolumn 'assets/paper/object_detection/ATSS2.png'%}


##  FCOS
Original paper ''FCOS: Fully Convolutional One-Stage Object Detection''. 
{% sidenote 1, 'Original FCOS paper see [here](https://arxiv.org/pdf/1904.01355.pdf). '%}

Key features:

1.**Proposal free and anchor free.**
FCOS is an one-stage object detector without anchors. 
The authors point out that the anchor is an heuristic which is hard to tune hyperparameters.
Without anchor, FCOS avoids a lot of hyperparameters tuning and complicated IOU calculation.

2.Since there is no anchor, FCOS uses keypoints (i.e. feature map pixels reproject to original image) to regress bounding box.
To overcome box overlap ambiguity, it arranges the regression size to different feature pyramids.

{% maincolumn 'assets/paper/object_detection/FCOS1.png'%}


3.It proposes a centerness branch to filter out low-quality predicted boxes whose locations are far away from the center of ground-truth objects. 

{% maincolumn 'assets/paper/object_detection/FCOS2.png'%}



##  RetinaNet
Original paper ''Focal Loss for Dense Object Detection''.
{% sidenote 1, 'Original RetinaNet paper see [here](https://arxiv.org/pdf/1708.02002.pdf). '%}

Key features:

1.One-stage (i.e. proposal-free) anchor-based detector.
Proposal-free means that it doesn't require bounding box proposals (candidates). 
Instead, it directly predicts from feature map (feature pixels).

2.It notices that the massive easy samples can dominate loss optimization (although the loss of easy samples is small, the sum of all easy sample losses is big).
It proposes the focal loss for addressing the positive and negative imbalance problem. 
Note that the imbalance problem is not new in object detection.
In fact, other one-stage detectors like SSD apply hard example mining and heuritsic thresholds to address this problem.

3.The authors also propose RetinaNet architecture.



##  YOLO
Original paper ''You Only Look Once: Unified, Real-Time Object Detection''.
{% sidenote 1, 'Original YOLO paper see [here](https://arxiv.org/pdf/1506.02640.pdf). '%}

YOLO has multiple versions.
I introduce the original version (i.e. YOLOv1).

Key features:

1.One-stage anchor-free detector.

2.Divide image to several cells. Predict and regress from feature map pixels directly.



##  SSD
Original paper ''SSD: Single Shot MultiBox Detector''.
{% sidenote 1, 'Original SSD paper see [here](https://arxiv.org/pdf/1512.02325.pdf). '%}

Key features:

1.One-stage anchor-based detector.

2.It predicts and regresses from different feature map levels.



##  Faster R-CNN
Original paper ''Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks''.
{% sidenote 1, 'Original Faster R-CNN paper see [here](https://arxiv.org/pdf/1506.01497.pdf). '%}

Key features:

1.Two-stage anchor-based detector.

2.It proposes region proposal network (RPN)

3.Another paper ''Feature Pyramid Networks for Object Detection'' proposes FPN based on Faster-RCNN.




##  Mask R-CNN
Original paper ''Mask R-CNN''.
{% sidenote 1, 'Original Mask R-CNN paper see [here](https://arxiv.org/pdf/1703.06870.pdf). '%}

Key features:

1.Two-stage anchor-based detector.

2.It is similar with Faster-RCNN, but it is with FPN and add a mask branch (i.e. semantic segmentation branch). 

3.It proposes RoI align technique instead of RoI pooling.

