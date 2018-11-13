---
layout: post
title: Point cloud registration
---

In this post, I mainly address the least square optimization problem in point cloud registration.

{% math %}
\mathbf{T^*} = \mathop{\arg\min}_{\mathbf{T^*}}\sum_{(p,q)\in\mathcal{K}} || \mathbf{p} - \mathbf{T}\mathbf{q} ||^2
{% endmath %}
where $$\mathbf{p}, \mathbf{q}$$ are the correspondence point pair in two point cloud pieces. $$\mathbf{T}$$ is rigid transformation matrix. 


Generally, there are two approaches to solve this problem (These two methods have been widely used in various non-linear optimization). 

I briefly summarize some important points here. {%sidenote 1, 'Please refer to [here](https://blog.csdn.net/kfqcome/article/details/9358853) for the detail of close-form derivation, and [here](https://zhuanlan.zhihu.com/p/33413665) for iterative solution.'%}


## Close-form solution 

The optimal translation is
{% math %}
\mathbf{t^*} = \mathbf{\bar q} - R\mathbf{\bar p}
{% endmath %}
where $$\mathbf{\bar q}, \mathbf{\bar p}$$ are the mean of correspondence points. $$R$$ is rotation matrix.

The optimal rotation is
{% math %}
R = VU^T
{% endmath%}
where $$V, U$$ is the SVD two component results of a matrix, which is composed by normalized correspondence points (see [derivation](https://blog.csdn.net/kfqcome/article/details/9358853) for the detail). 

Note that the SVD here is not applied for solving least square (the least square here is different from conventional format, check my previous post in [linear algebra section](../../linear_algebra/miscellaneous)).

Besides, the solution $$R = VU^T$$ may not be a true rotation matrix, because the derivation process is to solve the optimal orthodox matrix. Hence, $$R$$ may be a reflect matrix (the determinant of $$R$$ is -1). 
{%sidenote 2, 'the determinant of orthodox matrix $$X$$ either equals to 1 or -1 (this can be proved from $$XX^T=I$$ and det($$X$$)det($$X^T$$)=1), whereas det(rotation matrix)=1, det(reflect matrix)=-1'%}
In this case, rotation matrix should be modified as below, so that we can gurantee the det($$R$$)=1 (the reasons and details in above *derivation* link).

{% math %}
R = V 
\begin{pmatrix}
1 & & & & &  \\
 & 1 & & & &  \\
 & & ... & & &  \\
 & & & & 1 &  \\
  & & & &  & -1 \\
\end{pmatrix} 
U^T
{% endmath %}





## Iterative solution (Gauss Newton method)
There is another solution which is a more flexible and general format.

We represent the rigid transformation matrix into a vector $$ \xi = (\theta_x, \theta_y, \theta_z, x, y, z)$$ that collates rotation and translation components.

Suppose we have $$n$$ pairs correspondence points. We define the error function
{% math %}
f(\xi) = 
\begin{pmatrix}
p_1 - Tq_1 \\
p_2 - Tq_2 \\
... \\
p_n - Tq_n \\
\end{pmatrix}_{3n\times 1}
{% endmath %}
Note that we discard the last element in $$p-Tq$$ after calculation, and $$\xi$$ is included in $$T$$.
The original objective is $$\min ||f(\xi)||^2$$


Given a tiny disturbance $$\Delta \xi$$, and apply Taylor expansion. 
{% math %}
f(\xi + \Delta \xi) \approx f(\xi) + J(\xi) \Delta \xi
{% endmath %}
Where $$J$$ is the Jacobian matrix {%sidenote 3, 'first order is called Jacobian, and second order is called Hessian'%}.

Our target now is to find a best iteration direction $$\Delta \xi$$, which make the minimum $$\lVert f(\xi + \Delta \xi)\rVert^2 $$. In other word, we have
{% math %}
\Delta \xi^* = \mathop{\arg\min_{\Delta \xi}} \frac{1}{2} \lVert f(\xi)+J\Delta \xi \rVert^2
{% endmath %}
calculate derivation w.r.t. $$\Delta \xi$$, and make the derivation equals to 0. We have

{% math %} 
J(\xi)^T J(\xi) \Delta \xi = -J(\xi)^T f(\xi)
{% endmath %}
since $$\xi$$ is a initial guess, we can calculate $$\Delta \xi$$ (i.e. this equation tells us how to update $$\xi$$).

Note that $$J^T J$$ is Hessian matrix $$H$$ too.
{%sidenote 4, 'Pose graph optimization also uses this method to optimize. '%}

{% math %} 
H(\xi) \Delta \xi = -J(\xi)^T f(\xi)
{% endmath %}

Note that we cannot directly apply add to update $$\xi$$ because of the rotation component. One alternative way is to update $$T$$ as below (because $$\Delta \xi$$ is tiny, we have $$\sin\theta \approx \theta$$, $$\cos\theta \approx 0$$).
{% math %}
T_k \approx 
\begin{pmatrix}
1 & -\theta_z & \theta_y & x \\
\theta_z & 1 & -\theta_x & y \\
-\theta_y & \theta_x & 1 & z \\
0 & 0 & 0 & 1 \\
\end{pmatrix}
\times T_{k-1}
{% endmath %}


## Iterative solution (Levenberg-Marquardt method)

LM is similar with Newton method, but it controls a maximum bound for update step. Therefore, it is usually more robust than Newton method.

Please refer to [here](https://zhuanlan.zhihu.com/p/33413665) for the detailed introduction.

