---
layout: post
title: Reinforcement learning book review
---

I'll organize my notes in the same way with book "Reinforcement Learning: An Introducation" by Richard S. Sutton and Andrew G. Barto, Nov 5, 2017.

These notes are written for helping me quickly review relevant knowledge and some important insights. 

## Introduction

This chapter primarily introduce some basic concepts and history about RL.



## Part 1, Tabular Solution Methods

1. [Multi-armed Bandits](part1_muti_armed_bandits)
	
	This chapter mainly discuss the different methods for exploiting and explorating. It uses multi-armed bandits as a concrete example to compare the performance of those methods.

	Besides, it introduces a convergence condition about the action-value update.

	(Heads up: since in this multi-armed bandits example the decision is made based on the previous rewards, its action-value function is formulated from past rewards instead of future).

2. [Finite Markov Decision Processes](part1_finite_MDP)

	 This chapter gives the formal definition of RL, such as MDP, action-value function, state-value function, policy.

	 Moreover, it gives the Bellman equation to describe state-value and action-value including the optimal formulations as below.

	 {% math %}
	 v_*(s) = \max_a \mathbb{E}[R_{t+1}+\gamma v_*(S_{t+1})| S_t=s, A_t=a] \\
	 = \max_a \sum_{s^{'}, r} p(s^{'}, r|s, a)[r+\gamma v_*(s^{'})] \\ 

	 q_*(s,a) = \mathbb{E}[R_{t+1}+\gamma \max_{a^{'}} q_*(S_{t+1}, a^{'})| S_t=s, A_t=a] \\
	 = \sum_{s^{'}, r}p(s^{'}, r|s, a)[r+\gamma \max_{a^{'}}q_*(s^{'}, a^{'})]
	 {% endmath%}

	 (In this chapter, it focus on future rewards with a discount ratio.)

3. [Dynamic Programming](part1_DP)
	
	This chapter introduces how to find optimal policy by dynamic programming. This method only works when the environment is well-defined (i.e. model-based).
