---
layout: post
title: Bundle Adjustment
---

## Bundle Adjustment (BA)
Bundle adjustment is an important optimization in SLAM system. 
Unlike pose graph optimization which only optimize the poses, BA can optimize the 3D points and poses simultaneously for minimizing reprojection error.

Here is the objective function
{% math %}
\min \frac{1}{2} \sum_{j=1}^{m} \sum_{i=1}^{n} || u_i - \frac{1}{z_{i}} K (R_j P_i + t_j) ||^2
{% endmath %}
where $$u_i$$ is the image pixel. $$P_i$$ is 3D point in world coordinate (total n feature points). $$R_j, t_j$$ are the camera pose (total m frames). $$K$$ is camera intrinsic parameter. $$z_i$$ is the last element in $$P_i$$.
The optimized variables could be $$R_j, t_j, P_i$$.

If we write the objective in Lie Algebra, it will be
{% math %}
\min_{\xi, P} \frac{1}{2} \sum_{j=1}^{m} \sum_{i=1}^{n} || u_i - \frac{1}{z_{i}} K e^{\xi_j^{\wedge}} P_i ||^2
{% endmath %}

In fact, we can use a function $$h(\xi_j, P_i)$$ to represent $$\frac{1}{z_i} K e^{\xi_j^{\wedge}} P_i$$. If we want to undistort, those undistortion calculation can be also included in $$h(\xi_j, P_i)$$. 
{% sidenote 1, 'Recommend to read the slambook bundle adjustment in Chapter 10. '%}

Anyway, the objective can be 

{% math %}
\min_{\xi, P} \frac{1}{2} \sum_{j=1}^{m} \sum_{i=1}^{n} || u_i - h(\xi_j, P_i) ||^2
{% endmath %}

Obviously, there are many variables need to be optimized. We use $$x = [\xi_1, ..., \xi_m, P_1, ..., P_n]$$ to represent all variables.

we rewrite the objective as below

{% math %}
\min \frac{1}{2} ||f(x)||^2 = \frac{1}{2} \sum_{j=1}^{m} \sum_{i=1}^{n} ||e_{ij}||^2 = \frac{1}{2} \sum_{j=1}^{m} \sum_{i=1}^{n} || u_i - h(\xi_j, P_i) ||^2
{% endmath %}

In nonlinear optimization, we want to get the optimal $$\Delta x$$. So apply Taylor expansion, the objective will become

{% math %}
\min_{\Delta x} \frac{1}{2} ||f(x + \Delta x)||^2 = \frac{1}{2} \sum_{j=1}^{m} \sum_{i=1}^{n} || e_{ij} + F_{ij}\Delta \xi_j + E_{ij} \Delta P_i ||^2
{% endmath %}
where $$F_{ij}$$ is the jacobian of Lie algebra $$\xi_j$$, and $$E_{ij}$$ is the jacobian of world point $$P_i$$.

If we integrate all variable and apply GN or LM, we will still face the equation {% sidenote 1, 'If you have problem to understand, go to check the nonlinear optimization section. '%}
{% math %}
H \Delta x = g
{% endmath %}
$$H = J^TJ$$ in GN. While $$H = J^TJ + \lambda I$$ in LM.

In SLAM early stage, people think there are too many variables to optimize in real-time (i.e. those variables will generate a very huge $$J^T J$$ matrix in nonlinear optimization. It is prohibitive to calculate the $$\Delta x$$ by inversing $$J^T J$$). 

Later, researchers find there are some special structures in $$J^TJ$$. And $$J^T J$$ is a **sparse matrix** and there are special methods (so-called **marginalization**) to calculate $$\Delta x$$ quickly. 

For the math detail, go to read the slambook sparse and marginalization (稀疏性和边缘化) in Chapter 10.

