---
layout: post
title: Pose graph
---

Will write something here to introduce my understanding about pose graph optimization.

A intuitive tutorial can be found [HERE](https://blog.csdn.net/heyijia0327/article/details/47428553).

In pose graph optimization, the optimized variables are represented by nodes in graph.
And the error function is represented by edge.

We usually use G2O library to solve pose graph problems.
This library has implemented several template data structures and you can check the source code for the details.

Here I'd like emphasize the key points when you want to define yourself data structures.

Usually, you need to inherit the basic data structure from G2O library (e.g. BaseVertex, BaseUnaryEdge, BaseMultiEdge and etc.).
Then you should implement several key functions: 
{% sidenote 1, "There are some more functions that you need to implement in inheritance. I just give the core functions below. "%}

For Edge data structure:
1. Implement how to calculate the error function (i.e. evaluate error function).
2. Implement how to calculate the gradient of the error function w.r.t. its linked vertices.

For Vertex data structure:
1. Implement how to update the variable.
2. Implement how to set the initial value. 

