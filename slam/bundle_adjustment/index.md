---
layout: post
title: Bundle Adjustment
---

## Bundle Adjustment (BA)
Bundle adjustment is an important optimization in SLAM system. 
Unlike pose graph optimization which only optimize the poses, BA can optimize the 3D points and poses simultaneously for minimizing reprojection error.

Here is the objective function
{% math %}
\min \frac{1}{2} \sum_{i=1}^{n} || u_i - \frac{1}{z_i} K (RP_i + t) ||^2
{% endmath %}
where $$u_i$$ is the image pixel. $$P_i$$ is 3D point in world coordinate. $$R, t$$ are the camera pose. $$K$$ is camera intrinsic parameter. $$z_i$$ is the last element in $$P_i$$.
The optimized variables could be $$R, t, P_i$$.

If we write the objective in Lie Algebra, it will be
{% math %}
\min \frac{1}{2} \sum_{i=1}^{n} || u_i - \frac{1}{z_i} e^{\xi} P_i ||^2
{% endmath %}

For the details about the gradient calculation, please go to check [here](https://blog.csdn.net/luohuiwu/article/details/80748174).
