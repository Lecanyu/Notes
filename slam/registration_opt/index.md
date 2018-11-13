---
layout: post
title: Point cloud registration
---

In this post, I mainly address the least square optimization problem in point cloud registration.

{% math %}
\mathbf{T^*} = \mathop{\arg\min}_{\mathbf{T^*}}\sum_{(p,q)\in\mathcal{K}} || \mathbf{p} - \mathbf{T}\mathbf{q} ||^2
{% endmath %}
where $$\mathbf{p}, \mathbf{q}$$ are the correspondence point pair in two point cloud pieces. $$\mathbf{T}$$ is rigid transformation matrix


Generally, there are two approaches to solve this problem.


## Close-form solution



## Iterative solution



