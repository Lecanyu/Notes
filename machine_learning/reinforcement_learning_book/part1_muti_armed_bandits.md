---
layout: post
title: Multi-armed Bandits
---

## Action-value estimation

The action-value function usually can be updated according to below equation:

{% math %}
Q_{new}(A) \gets Q_{old}(A) + \alpha_{n}[Q_{target}(A) - Q_{old}(A)]
{% endmath %}
The learning rate $$\alpha_n$$ here has been treated as the function w.r.t. update times $$n$$.

$$Q_{new}(A)$$ can converge when below two conditions are satisfied.

{% math %}
\sum_{n=1}^\infty \alpha_n = \infty \\
\sum_{n=1}^\infty \alpha_n^2 < \infty
{% endmath %}

The first condition is required to guarantee that the steps are large enough to eventually overcome any initial conditions or random fluctuations. 

The second condition guarantees that eventually the steps become small enough to assure convergence.

Note that $$\alpha$$ usually is set to a constant in many practical RL problems, which violates the second condition. Hence, the $$Q_{new}$$ won't converge. However, this actually can be a desirable property in highly nonstationary environment (i.e. the true state-value can be changed after sampling each time).


## Several exploration strategies

1. $$\epsilon$$-greedy

2. Upper confidence bound (UCB)
	
	This method is adopted in recent AlphaGo monte carlo tree search. 

	The action will be taken according to below rule

	{% math %}
	a = \mathop{\arg\min}_{a} [Q_t(a) + c \sqrt \frac{\ln t}{N_t(a)}]
	{% endmath %}
	where $$Q_t(a)$$ is the action-value result at time step $$t$$. $$N_t(a)$$ denotes the number of times that action $$a$$ has been selected prior to time $$t$$.

	The idea of this action selection is that each time $$a$$ is selected, the uncertainty of action $$a$$ is presumbly reduced, and thus the probability of selecting $$a$$ will decrease.

3. Optimistic initial values

	Setting a larger initial estimates can encourage exploring automatically in some special cases.

4. Gradient bandit algorithm

	Directly define a numerical preference for each action selection.





 

