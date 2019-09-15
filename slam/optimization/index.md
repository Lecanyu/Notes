---
layout: post
title: Nonlinear Optimization
---

## Nonlinear optimization
The nonlinear optimization can be written as 
{% math %}
x = \arg \min_x F(x)	
{% endmath %}
where $$F(x)$$ is a nonlinear function w.r.t. $$x$$. Since the $$F(x)$$ can be extremely complicated, we may not be able to explicitly figure out the analytical solution of 
$$\frac{\partial F}{\partial x} = 0$$.
Instead of the analytical solution, we usually apply iteration methods to optimize the objective, even though it may fall into local optimal.

There are four iteration methods: Gradient Descent, Newton method, Gauss-Newton method, Levenberg–Marquardt (LM). 

### Gradient Descent
Given a step size hyperparameter $$\alpha$$, the $$x$$ update rule is 
{% math %}
x = x - \alpha \frac{\partial F}{\partial x}	
{% endmath %}
This optimization strategy (but a variant SGD) has been widely adopted in various neural network update.


### Newton method
Newton method try to solve this objective
{% math %}
\Delta x = \arg \min_{\Delta x} F(x + \Delta x)	
{% endmath %}
It applies the Taylor expansion {%sidenote 1 'Note that the Jaconbian J(x) and Hessian H(x) are first and second derivation of F(x) w.r.t. x'%}
{% math %}
F(x + \Delta x)	= F(x) + J(x)\Delta x + \frac{1}{2} \Delta x^T H(x) \Delta x
{% endmath %}
To minimize above equation, we calculate the gradient w.r.t. $$\Delta x$$ and make it equal to 0.
We have
{% math %}
\Delta x = - H(x)^{-1}J(x)^T
{% endmath %}
However, the second derivation is usually expensive to calculate

### Gauss-Newton
Gauss-Newton solve the problem from least-square perspective.
$$F(x)$$, which is a scalar number, usually come from $$\frac{1}{2}f(x)^Tf(x)$$, where $$f(x)$$ is a vector.
This method apply taylor expansion on $$f(x)$$ to first derivation.
We have
{% math %}
f(x + \Delta x)	= f(x) + J(x)\Delta x
{% endmath %}
So
{% math %}
F(x+\Delta x) = \frac{1}{2} f^T(x + \Delta x) f(x + \Delta x) = \frac{1}{2}f^T(x)f(x) + \Delta x^T J^T(x)f(x) + \frac{1}{2} \Delta x^T J^T(x) J(x) \Delta x
{% endmath %}
Calculating the gradient w.r.t. $$\Delta x$$, we have	{%sidenote 2 'Note that the Jaconbian J(x) and Hessian H(x) are first and second derivation of f(x) w.r.t. x. This is different with Newton method.'%}
{% math %}
\Delta x = - (J^T(x)J(x))^{-1}J(x)^Tf(x)
{% endmath %}

However, Gauss-Newton method still has problems. 
First, the $$J^T(x)J(x)$$ needs to be invertible , but this is not guaranteed. 
Second, if $$\Delta x$$ is big, the above Taylor expansion is a bad approximation. 
To solve these two problems, LM method is invented.

### Levenberg–Marquardt (LM)
The main motivation of LM is to control the size of update step $$\Delta x$$.
It optimizes the below objective
{% math %}
\min \frac{1}{2} ||f(x) + J(x)\Delta x||^2 \\
s.t. ||\Delta x||^2 \le \mu
{% endmath %}
where $$\mu$$ is the confidence interval, which will be updated for each optimization step.

The update rule:

For each optimization step, we calculate
{% math %}
\rho = \frac{f(x+\Delta x) - f(x)}{J(x)\Delta x}
{% endmath %}
If $$\rho$$ is closed to 1, the local linear is guaranteed and the approximation is good.
If $$\rho$$ is larger than 1, the actual decrease is more than the approximation which means the $$f(x)$$ is accelerated decreasing. And we should set a bigger confidence interval $$\mu$$ to speed decrease.
If $$\rho$$ is smaller than 1, the actual decrease is less than the approximation which means the $$f(x)$$ is entering a flatted area. And we should set smaller confidence interval $$\mu$$.

The above constrained objective can be converted to dual space by applying Lagrange multipler.
{% math %}
\max_{\lambda} \min_{\Delta x} \frac{1}{2} ||f(x) + J(x)\Delta x||^2 + \lambda (\Delta x^T \Delta x - \mu)
{% endmath %}
The optimal $$\Delta x$$ satisfies 
{% math %}
(J^T(x)J(x) + \lambda I)\Delta x = J^T(x)f(x)
{% endmath %}
If $$\lambda$$ is big which means $$\Delta x^T \Delta x - \mu$$>0 and $$\Delta x$$ is over the confidence interval, the $$\Delta x$$ will be $$\frac{1}{\lambda}J^T(x)f(x)$$ (i.e. Gradient Descent update).
If $$\lambda$$ is small, then the update $$\Delta x$$ is like Gauss-Newton $$J^T(x)J(x) \Delta x = J^T(x)f(x)$$.



### A SGD variant: Adam
The LM method adaptively determines the update step. In gradient descent, there is also a famous adaptive method, called Adam, which has been widely used in neural network optimization.
Here is Adam update rule {%sidenote 3 'Check [here](https://www.cnblogs.com/wuliytTaotao/p/11101652.html) for others introduction.'%}:
{% math %}
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t \\
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2 \\
\hat m_t = \frac{m_t}{1-\beta_1^t} \\
\hat v_t = \frac{v_t}{1-\beta_2^t} \\
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat v_t} + \epsilon} \hat m_t
{% endmath %}
Usually, $$\beta_1 = 0.9, \beta_2 = 0.99, \eta=0.001$$, $$\eta$$ is learning rate. 

The first two equations are momentum update (i.e. moving average). 
The third and fourth equations are bias corrections. Because $$m_t$$ is underestimated at the beginning (i.e. $$m_t = 0.1 g_t$$).
$$\beta_1^t, \beta_2^t$$ will gradually decrease to 0 as the optimization proceed.
The last equation is the parameter update rule. 
The learning rate $$\eta$$ will be adjusted by $$\frac{1}{\sqrt{\hat v_t} + \epsilon}$$. 
Obviously, at the beginning $$\hat v_t$$ is small ($$\hat v_t$$ is used to accumulate the gradient). But after many iterations, $$\hat v_t$$ could be big, and thus the learning rate will be adjusted to 0.


