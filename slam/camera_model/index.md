---
layout: post
title: Camera Model
---

## Pinhole Camera Model
Given a 3D point $$P = \begin{pmatrix} x\\ y\\ z \end{pmatrix}$$ in world coordinate, we can calculate the corresponding image pixel $$Q = \begin{pmatrix} u\\ v\\ 1 \end{pmatrix}$$ (homogeneous coordinate):
{% math %}
zQ = K(RP+t) \\ 
{% endmath %}
where $$K = \begin{pmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{pmatrix}$$ is internal camera parameters.

$$R,t$$, which represent the camera rotation and translation in world coordinate, are external camera parameters.


## Binocular Camera Model
Here, I also put the introduction of binocular camera model. 
{% maincolumn 'assets/slam/binocular_camera_model.png'%}



## Fundamental and Essential Matrix
Fundamental and essential matrix are two important concepts in epipolar constraint. It has been widely used in various monocular 3D vision.
Here are the derivations.

{% maincolumn 'assets/slam/epipolar_constraint.png'%}
Given two correponded feature points $$p_1, p_2$$ in two images ($$P$$ is the 3D point in world coordinate) and assuming the first camera is on orginal coordinate, we have below equations.
{% math %}
z_1p_1 = K_1P \\ 
z_2p_2 = K_2(RP+t)
{% endmath %}
Since $$z_1$$ is a scalar and $$p_1 = \begin{pmatrix} u_1\\ v_1\\ 1 \end{pmatrix}$$. We can put $$z_1$$ into $$p1$$ for simplicity.
$$p_1^{'} = z_1 p_1 = \begin{pmatrix} z_1 u_1\\ z_1 v_1\\ z_1 \end{pmatrix}$$.
$$p_2^{'} = z_2 p_2 = \begin{pmatrix} z_2 u_2\\ z_2 v_2\\ z_2 \end{pmatrix}$$.
So we have
{% math %}
p_1^{'} = K_1P \\ 
p_2^{'} = K_2(RP+t)
{% endmath %}
We have below equation after combining them.
{% math %}
K_2^{-1} p_2^{'} = RK_1^{-1}p_1^{'}+t
{% endmath %}
We left cross product $$t$$ on both sides, and the equation becomes
{% math %}
[t]_{\times} K_2^{-1} p_2^{'} = [t]_{\times} RK_1^{-1}p_1^{'}
{% endmath %}
where $$[t]_{\times}$$ is the cross product matrix.
Since $$[t]_{\times} K_2^{-1} p_2^{'}$$ is a vector and it is vertical vector $$K_2^{-1} p_2^{'}$$.
We left multiply $$(K_2^{-1} p_2^{'})^T$$ on both sides. Then we have 
{%sidenote 1 'Note that the scale in this estimation is undetermined. Because you can multiply any scalar number on essential matrix without violate the equation. This problem leads to the depth estimation is undetermined too. So monocular vision cannot calculate the exact scale information.'%}

{% math %}
0 = p_2^{'T} \underbrace{K_2^{-T} \overbrace{ [t]_{\times} R}^{\text{Essential matrix}} K_1^{-1}}_{\text{Fundamental matrix}} p_1^{'}
{% endmath %}
If we already have interal camera parameters $$K_1, K_2$$, we can calculate the essential matrix from the a group of feature correspondences and then decomposite essential matrix to the external parameter $$R, t$$.
If we don't have interal camera parameters, then we need to calculate the fundamental matrix and figure out the interal and external parameters at the same time.



## Homography matrix
Sometimes, the fundamental or essential matrix could be ill-posed (e.g. no translation $$t=0$$ or all correspondence points are on a plane). The essential or fundamental decomposition could be degenerated.

In this case, we build a new model to calculate interal and external parameters.
{% maincolumn 'assets/slam/homography_model.png'%}
Given a plane in world coordinate (usually we can assume the first camera coordinate is world coordinate), the plane can be formulated as below
{% math %}
n^T P + d = 0
{% endmath %}
where $$n$$ is the normal vector of plane. $$P$$ is a point on the plane. $$d$$ is the vertical distance between original coordinate and plane.

The corresponded feature points on two images should satisfy the below equation
{% math %}
p_2^{'} = K_2(RP+t)
{% endmath %}
Since $$-\frac{n^T P}{d} = 1$$, we can formulate a new equation
{% math %}
p_2^{'} = K_2(RP-t\frac{n^T P}{d}) = K_2(R-\frac{tn^T}{d})P = K_2(R-\frac{tn^T}{d})K_1^{-1}p_1^{'}
{% endmath %}
That is 
{%sidenote 2 'Note that the complete homography matrix should contain the depth information $$z_1, z_2$$, like $$\frac{z_1}{z_2}K_2(R-\frac{tn^T}{d})K_1^{-1}$$.'%}

{% math %}
p_2^{'} = \underbrace{K_2(R-\frac{tn^T}{d})K_1^{-1}}_{\text{Homography matrix}}p_1^{'}
{% endmath %}
If we have a group of feature correspondences, then we can estimate homography matrix and decompose it to $$R, t, n, d, K_1, K_2$$.



## Trianglation (depth estimation)
There are several methods to estimate the depth of points, when we get the intrinsic and extrinsic camera parameters. 
Here I'd like to introduce one of them.
First, we have
{% math %}
z_1p_1 = K_1P \\ 
z_2p_2 = K_2(RP+t)
{% endmath %}
Combine them and we can get
{% math %}
z_2K_2^{-1}p_2 = z_1RK_1^{-1}p_1 + t
{% endmath %}
Since $$K_1, K_2$$ are already known, we use $$p_1^{'} = K_1^{-1}p_1, p_2^{'} = K_2^{-1}p_2$$ for simplicity.

So we have 
{% math %}
z_2p_2^{'} = z_1Rp_1^{'} + t
{% endmath %}
Multiple $$[p_2^{'}]_{\times}$$ in both sides. We have
{% math %}
0 = z_1[p_2^{'}]_{\times}(Rp_1^{'} + t)
{% endmath %}
Since $$p_1^{'}, p_2^{'}, R, t$$ are already known, we can solve $$z_1$$ now. Then we can calculate $$z_2$$.



