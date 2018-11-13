---
layout: post
title: Reinforcement learning
---

## The difference of several concepts (terminologies)

1. On-policy vs Off-policy
	
	Off-policy is the training target of a policy is different from the behavior policy like $$\epsilon$$-greedy (more exploration).

	On policy is the training target of a policy is exact the same with the behavior policy. (less exploration).

	On-policy can usually converge faster than off-policy.


2. Model-based vs Model-free

	Model-based: there are descriptions about the environment, such as the probabilistic distribution of rewards.

	Model-free: No explicit descriptions about the environment where the agents operate. 

3. Exploration vs Exploitation

	Exploration: explore unknown area. It may be beneficial in the long term. Typically, exploration is necessary when there is always uncertainy about the accuracy of the action-value estimates.

	Exploitation: greedily choose current best action. It is usually a local optimal action.


4. Value function based methods vs Evolutionary methods
	
	Value function based methods are trying to explore the value of a particular state, and then take advantage of the value function to take an action.

	Evolutionary methods are simply ergodic strategies. It attempts every possible policy and evaluates its rewards. Hence, it only works when policy space is sufficiently small, or can be structured (i.e. the good policy can be easy to find).

5. Reward vs Value
	
	reward is immediate, but value need to evaluate the reward in the future.

6. Temporal difference (TD) learning
	
	TD learning has below format 

	{% math %}
	V(s) \gets V(s) + \alpha [r(s) + V(s^{'}) - V(s)]
	{% endmath %}		
	where $$s$$ is current state, $$s^{'}$$ is the next state; $$V(\cdot)$$ is value function. $$\alpha$$ indicates the learning rate (update rate). 

	This update rule is temporal-difference learning, because its changes are based on a difference, $$V(s^{'}) - V(s)$$, between estimates at two different times.

	The renowed Q-learning is an off-policy TD learning method.

7. Evaluative feedback vs instructive feedback
	
	Purely evaluative feedback indicates how good the action taken was, but not whether it was the best or the worst action possible.

	Purely instructive feedback, on the other hand, indicates the correct action to take, independently of the action actually taken.

	In one word, evaluative feedback depends entirely on the action taken, whereas instructive feedback is independent of the action taken. (i.e. evaluative feedback is based on partial information, whereas instructive feedback can only be made under full information).

8. Three fundamental classes of methods for solving finite markov decision problems.

	**Dynamic programming**

	This kinds of methods are well developed mathematically, but require a complete and accurate model of the environment

	**Monte Carlo methods**

	Monte Carlo methods don’t require a model and are conceptually simple, but are not well suited for step-by-step incremental computation

	**Temporal-difference learning**

	Temporal-difference methods require no model and are fully incremental, but are more complex to analyze.


