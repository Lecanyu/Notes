---
layout: post
title: Expectation Maximization
---

Expectation Maximization (a.k.a. EM algorithm) is popular in estimating parameters in model which contains hidden random variable.

I give a specific example which is simple and intuitive to explain the principle here. 

For more advanced techniques about EM algorithm, you may refer to <统计学习方法>


## 硬币模型例子
假设有三枚硬币A、B、C，它们正面朝上的概率是$$\pi, p, q$$，按如下规则掷硬币：先掷硬币A，如果A正面朝上则选择硬币B进行投掷，如果A反面朝上则选择硬币C进行投掷，最后记录B或者C的投掷结果作为输出。
这样独立重复地进行n次实验，可得到一系列观测结果$$Y$$(比如$$Y=1101001$$，1表示正面朝上)。

假如只能观察到硬币最后的投掷结果，而不知道投掷过程中的隐变量，现在想通过观测结果估计硬币模型的参数(即$$\pi, p, q$$)，该如何进行？


该问题可用概率模型进行形式化描述：
{% math %}
P(y|\theta) = \sum_z P(y, z|\theta) = \sum_z P(z|\theta)P(y|z,\theta)
{% endmath %}
这里$$z$$表示隐变量，$$y$$表示模型输出结果，$$\theta=(\pi, p, q)$$是模型参数。
该模型符合直觉，第一项$$P(z|\theta)$$的含义是当已知参数时，隐变量取某值的概率。
第二项$$P(y|z,\theta)$$的含义是当隐变量和模型参数确定时，产生最终输出的概率。


在硬币模型例子中，分别考虑$$z$$的正反两种取值
{% math %}
P(y|\theta) = \sum_z P(z|\theta)P(y|z,\theta) = \pi p^y (1-p)^{(1-y)} + (1-\pi) q^y (1-q)^{(1-y)}
{% endmath %}

对于一系列的某观测结果发生的概率是
{% math %}
\prod_{i=1}^n P(y_i|\theta) = \prod_{i=1}^n \pi p^{y_i} (1-p)^{(1-y_i)} + (1-\pi) q^{y_i} (1-q)^{(1-y_i)}
{% endmath %}

进行极大似然估计，取对数可以得到下面的目标函数
{% math %}
f(\theta) = \arg\max_{\theta} \log \prod_{i=1}^n P(y_i|\theta) => f(\pi, p, q) = \arg\max_{\pi, p, q} \sum_{i=1}^n \log (\pi p^{y_i} (1-p)^{(1-y_i)} + (1-\pi) q^{y_i} (1-q)^{(1-y_i)})
{% endmath %}

一个自然的想法是，对目标函数关于各个参数求偏导，并令偏导数为0即可。
不过这里
{% math %}
\frac{\partial f(\pi, p, q)}{\partial \pi} = \sum_{i=1}^n \frac{p^{y_i} (1-p)^{(1-y_i)} - q^{y_i} (1-q)^{(1-y_i)}}{\pi p^{y_i} (1-p)^{(1-y_i)} + (1-\pi) q^{y_i} (1-q)^{(1-y_i)}}
{% endmath %}
是否等于0不受$$\pi$$控制，也就是说$$\pi$$没有最优解析解，对于这个目标函数通常需要用初值迭代的方式进行。


### EM算法迭代

E步:用当前的参数估计值来估计隐变量的概率分布
{% math %}
P(Z|Y, \theta_i)
{% endmath %}
放到这个硬币模型中来说就是在给定观测$$Y=y_i$$和参数$$\theta_i$$的情况下，计算$$P(Z=z_i|Y=y_i, \theta_i)$$即第二次掷的是硬币B还是硬币C的概率。


M步:计算$$\log P(Y,Z|\theta)$$关于估计得到的隐变量的期望，使该期望最大化，即
{% math %}
\arg\max_{\theta} \sum_{Z} P(Z|Y, \theta_i) \log P(Y, Z|\theta)
{% endmath %}

E步很符合直觉，而M步初看之下似乎是反直觉的。
为什么要计算$$\log P(Y,Z|\theta)$$关于估计得到的隐变量的期望，这样一个奇怪的东西。

直觉上讲，E步估计了隐变量的概率分布之后，
直接用这个隐变量结果来让$$P(Y|Z, \theta)$$最大化不就行了吗？
（半监督学习中的伪标签策略就是这样做的）
{% sidenote 1, '注意这种是对EM算法最常见的错误理解，我之前也是这么认为的。 '%}

但这种理解是自训练，而并不是EM算法。
M步中奇怪的$$\log P(Y,Z|\theta)$$背后是有数学原因的。


### EM算法的导出

对于一个包含隐变量的概率模型，目标是进行观测数据对参数的极大似然估计

{% math %}
\arg\max_{\theta} L(\theta) = \log P(Y|\theta) = \log \sum_{Z} P(Z|\theta)P(Y|Z, \theta)
{% endmath %}

由于这里含有求和（或者积分）的对数，这给目标函数的优化带来了困难。
而EM算法并不直接优化上式，而是希望逐渐的使得$$L(\theta)$$增大（迭代式的优化）
即 $$L(\theta_{i+1}) - L(\theta_i) > 0$$（第i+1次迭代比第i次大）,
为了视觉上容易区分，下面的推导用$$\theta$$代替$$\theta_{i+1}$$
{% sidenote 1, '这里的推导用了Jensen不等式: $$\log \sum_i k_i y_i \geq \sum_i k_i \log y_i  $$，其中$$k_i \geq 0, \sum_i k_i = 1$$。 '%}

{% math %}
L(\theta) - L(\theta_i) = \log[\sum_Z P(Y|Z, \theta)P(Z|\theta)] - \log P(Y|\theta_i) \\
= \log[ \sum_Z P(Z|Y, \theta_{i}) \frac{P(Y|Z, \theta)P(Z|\theta)}{P(Z|Y, \theta_{i})} ] - \log P(Y|\theta_i) \\
\geq \sum_Z P(Z|Y, \theta_{i}) \log \frac{P(Y|Z, \theta)P(Z|\theta)}{P(Z|Y, \theta_{i})} - \log P(Y|\theta_i) \\
= \sum_Z P(Z|Y, \theta_{i}) \log P(Y|Z, \theta)P(Z|\theta) - \sum_Z P(Z|Y, \theta_{i}) \log P(Z|Y, \theta_{i}) - \log P(Y|\theta_i)
{% endmath %}

我们希望使得每次迭代$$L(\theta)$$尽可能大，因此我们可以最大化$$L(\theta) - L(\theta_i)$$。
注意到上面推导最后一行只有第一项与$$\theta$$有关，因此问题等价于求解
{% math %}
\arg\max_{\theta} L(\theta) - L(\theta_i) = \arg\max_{\theta} \sum_Z P(Z|Y, \theta_{i}) \log P(Y|Z, \theta)P(Z|\theta) \\
= \arg\max_{\theta} \sum_Z P(Z|Y, \theta_{i}) \log P(Y, Z|\theta)
{% endmath %}
这个就是上面的计算$$\log P(Y,Z|\theta)$$关于估计
得到的隐变量的期望$$\arg\max_{\theta} \sum_{Z} P(Z|Y, \theta_i) \log P(Y, Z|\theta)$$

至此，我们说明了为什么EM算法的M步要计算关于隐变量的期望最大化。
从上面的推导也不难发现，EM的本质是不断的迭代提升$$L(\theta)$$的下界来近似最大化的。



