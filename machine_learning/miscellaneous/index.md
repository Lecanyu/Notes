---
layout: post
title: Miscellaneous
---

Some basic machine learning knowledge which are frequently applied in various research (also often asked in interview). I wrote them down for review.

## Pareto optimality
It is a resource allocation state in which it is impossible that reallocate the resources so as to improve an individual situation without making other individuals worse off.  

It is a minimum notion of efficiency but unnecessarily lead to desire results since it doesn’t take fairness and equality into account.


## Pareto improvement
It means we can reallocate the limited resources to make some individuals better off without making any other individuals worse off.


## Precision-recall estimation
{% math %}
Recall = \frac{TP}{TP+FN}, \quad
Precision = \frac{TP}{TP+FP}
{% endmath %}

False positive (FP): classify the negative class into positive category.

False negative (FN): classify the positive class into negative category.

## Average precision (AP and mAP)
AP is a common metric in various object detection paper. 
It is related with recall and precision. 
Conceptually, it is the area of recall-precision curve. See below picture.
{% maincolumn 'assets/machine_learning/AP.png'%}

But for convenience, people usually approximately estimate this area by interpolation. 
Check [here](https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173) for the specific algorithm. 


## Intersection over union (IoU)
This metric has been widely used on object detection. 
It is used for measuring how accurate the predicted bounding box is.
{% maincolumn 'assets/machine_learning/IoU explanation.png'%}



## Bias and variance
Bias is used to describe how close it is between the prediction and true value.

Variance is used to describe how concentrated the predictions are.

Overfitting usually lead to high variance since it is trained to fit the noises in training data.

Underfitting usually lead to high bias since it is unable to capture enough correlation between input and target.

## Classification and regression
The training target in classification is discrete.

The training target in regression is continuous.


## Supervised learning, unsupervised learning, semi-supervised learning
The training data is labeled in supervised learning like classification

The training data is unlabeled in unsupervised learning like clustering, generative adversarial network, EM algorithm, PCA and etc.


## Off-policy and on policy
Off-policy is the training target of a policy is different from the behavior policy like 𝜀-greedy (more exploration)

On policy is the training target of a policy is exact the same with the behavior policy. (less exploration)

On policy can converge faster than off-policy.

## Feature selection (how to select feature from massive statistical data)
Filtering: according to the feature variance

Wrapper: we can randomly pick some features to train and evaluate the result

Embedded: we can use a machine learning method to train first and then check how important those features are.

## Generative model
All types of generative models aims at learning the true distribution of training data so as to generate new data point with some variations. But it is not always possible to learn the exact data distribution.

It belongs to unsupervised learning (we don't need to label the training data).

Two types of generative models:
+ Variational autoencoder (VAE)
+ Generative Adversarial Networks (GAN) 
{% sidenote 1, "check [here](https://towardsdatascience.com/generative-adversarial-networks-explained-34472718707a) and [paper](https://arxiv.org/pdf/1406.2661.pdf) for detailed introducation. "%}
 - *The generator network and discriminator network (evaluator).*
{% maincolumn 'assets/machine_learning/generative_adversarial_network.png'%}
{% maincolumn 'assets/machine_learning/generative_adversarial_network_2.png'%}

## P, NP, NP-Complete, NP-hard
P is the problem that we can find a solution which can be finished in polynomial time. 
{% marginfigure 'mf-id-whatever' 'assets/machine_learning/P_NP.png' 'The relationship between P, NP, NP-hard and NP-Complete. '%}

NP is the problem that we may not be able to find a polynomial time complexity solution but we can validate if a specific solution is correct or not within polynomial time.

NP-hard is a more general problem that some NP problems can reduce to within polynomial time. Once a NP-hard problem is solved, all such reducible NP problems will be solved. (Note: Sometimes, after NP reduce to NP-hard, this NP-hard problem may not be validated within polynomial time).
The global composition in jigsaw puzzle solving can be seen as a SAT problem variant, but it cannot be validated within polynomial time. So it belongs to NP-hard.

NP-complete is overlap between NP-hard and NP, which means they are reduced from NP and still can be validated within polynomial time.

SAT (satisfiability) problem: if we can find a Boolean assignment that make a system output true. For example, we cannot find such assignment for  $$\neg p \land p$$, but we can find for $$ p \land q $$.

Variant: 2-SAT and 3-SAT. See this intuitive [example](https://www.zhihu.com/question/55516280/answer/145138234) (过年了，正打算烧年夜饭 blabla...)

Problem Reduce: this means we can convert a problem into a more general (usually more difficult) problem. For example, problem A: find the minimum element in an array. Problem B: sort the whole array. We can say A can be reduce to B, since if we can solve the problem B, problem A will be solved trivially.

## The advantage of max pooling layer
Reduce the number of parameters. To avoid over-fitting

Increase the perceptual field

Extract the main features.

## Convolutional Neural Network
{% maincolumn 'assets/machine_learning/CNN_filter_calc.png'%}

## Batch normalization
{% maincolumn 'assets/machine_learning/batch_normalization.png'%}

Suppose input= [batch, height, width, depth]. 
If we use axes= [0,1,2] to calculate (e.g. mean, var = tf.nn.moments(input, axes=[0, 1, 2])), then the output will be a 1-D vector with size=depth. 
If we use axes=[0] to calculate, then the output will be a 3-D vector with size=[height, width, depth]. The picture below demonstrates this 3-D vector case.

{% maincolumn 'assets/machine_learning/batch_normalization_ex.png'%}


## Recurrent Neural Network (RNN)
An tutorial and introduction about [RNN](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/)

{% maincolumn 'assets/machine_learning/RNN.png'%}

## Hidden Markov Model 
An specific dice [example](https://www.zhihu.com/question/20962240) will help to understand.

The basic elements:
Transition matrix, emission matrix, initial state.

Three basic problems:
1. Given the HMM model, how to calculate the probability of an observation. (forward algorithm)
2. Given the HMM model and an observation, how to estimate which the most possible hidden state sequences are. (Viterbi algorithm, dynamic programming)
3. Given the observation data, how to estimate the model parameters. (learning problem, Baum–Welch algorithm)

Note:
In the learning problem, 
the target of forward algorithm is to calculate the $$\alpha_i(t) = P(X_{t=i}, y_1,y_2, ..., y_t)$$.
The target of backward algorithm is to calculate the $$\beta_i(t) = P(y_{t+1}, y_{t+2}, ..., y_T|X_{t=i})$$.
The target of forward-backward algorithm is to calculate the $$\gamma_i(t) = P(X_{t=i}|y_1, y_2, ..., y_T)$$.

For the details, please check [here](https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm)


## EM (Expectation-Maximization) algorithm
An intuitive [tutorial](https://www.youtube.com/watch?v=REypj2sy_5U).

EM algorithm is an unsupervised learning.

## Maximum likelihood vs Maximum A Posterior (MAP)
$$ P(X|Y) = \frac{P(Y|X) * P(X)}{P(Y)} $$ <=> $$ Posterior = \frac{Likelihood * Prior}{Evidence} $$.

Usually given the training data $$D$$, our target is to maximize the posterior $$P(\theta|D)$$, where $$\theta$$ is the model parameters. 
To do that, we usually have two approaches:

1. Maximum likelihood.
Since $$P(\theta|D)=\frac{P(D|\theta)P(\theta)}{P(D)}$$, if we can assume the prior is a constant, then $$ \max P(\theta|D) = \max P(D|\theta)$$

2. Maximum A Posterior (MAP).
Sometimes, we may not be able to make such assumption that the prior is a constant. Instead, prior may satisfy an unknown distribution. In this case, the target is  $$P(\theta|D) =  \mathop{\arg\min}_{\theta} P(D|\theta)P(\theta)$$. Since every data is generated independently, we have $$ P(\theta|D) =  \mathop{\arg\min}_{\theta} \Pi_i^n P(D=x_i|\theta)P(\theta) $$.


## A systematic probabilistic graph model course
https://ermongroup.github.io/cs228-notes/ 

## Bayes rule and Bayes network
Bayes rule is widely used on various machine learning problems. Some basic math/probabilistic knowledge need to be clarified.

The general Bayes rule for joint distribution:
{% math %}
P(x_1, x_2, ..., x_n) = P(x_1|x_2, ...,x_n)*P(x_2|x_3, ..., x_n)* ... * P(x_{n-1}|x_{n}) * P(x_{n})
{% endmath %}

If we have a bayes network which models the problem, we can simplify the calculation. For example, given a simple network as below, we have 
{% maincolumn 'assets/machine_learning/bayes_net1.png'%}
{% math %}
P(A, B, C) = P(A)*P(B|A)*P(C|B)
{% endmath %}
This can work because {%m%} P(C|B) = P(C|A, B) {%em%} (when we know variable B, the variable A and C can be seen independent. But if B is unknown, A and C are dependent. This is because we can use the medium variable B to represent the dependence between A and C.)

From this network, we can have some other property like 
{%math%} P(C|A) = \int P(C|B)P(B|A) \mathrm{d}B {%endmath%}
Because {%m%} P(C|B) = P(C|A, B), P(C|B)P(B|A) = P(C|A, B)*P(B|A) = \frac{P(A, B, C)}{P(A)} {%em%} and {%m%} \int P(A,B,C) \mathrm{d}B = P(A,C) {%em%} 


## Several concepts need to be distinguished
1. Bayesian Network
2. Markov Random Field
3. Conditional Random Field
4. Markov Chain
5. Hidden Markov Model
6. Markov Decision Process

## How to simulate a random number which satisfy a probabilistic distribution 
1. Inverse transform method
2. Acceptance rejection method

Check [here](http://blog.codinglabs.org/articles/methods-for-generating-random-number-distributions.html)

## Why divide n-1 to get unbiased variance estimation in sampling data
http://blog.sina.com.cn/s/blog_c96053d60101n24f.html 

## A good systematic Chinese tutorial for machine learning
https://nndl.github.io/ 

The summary of [math](https://nndl.github.io/chap-%E6%95%B0%E5%AD%A6%E5%9F%BA%E7%A1%80.pdf) part is a good material for reviewing the math background.


## Information theory
Refer to [here](https://nndl.github.io/chap-%E6%95%B0%E5%AD%A6%E5%9F%BA%E7%A1%80.pdf) for detailed introducation.

Encode length for a random variable $$X=x$$, $$I(x)= -\log p(x)$$.

Entropy: the average length for optimal encoding the whole of random variable X.  $$H(x) = -\sum_x p(x) \log p(x)$$. 
The more stochastic variable is, the larger entropy is.

{% math %}
\mbox{Joint entropy:  } H(x, y) \\
\mbox{Conditional entropy:  } H(x|y) \\
H(x|y) = H(x,y) - H(y)
{% endmath %}

Cross-entropy: the average encoding length when we use distribution $$q$$ to encode information $$x$$ whose real distribution is $$p$$. 
{% math %}
H(p,q)= -\sum_x p(x) \log q(x)
{% endmath %}
Obviously, $$H(p,q)=H(p)$$, when $$p(x)=q(x)$$. We have minimum $$H(p,q)$$, when $$p(x)=q(x)$$.

Kullback-Leibler divergence (KL-divergence): 
{% math %}
H(p||q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = H(p,q) - H(p) >= 0
{% endmath %}
The meaning of KL-divergence is very similar with cross-entropy. 
Both of them measure how different between distribution $$p(x)$$ and $$q(x)$$. So when $$p(x)$$ is given, the optimization process of $$H(p||q)$$ and $$H(p,q)$$ is the same.
Comparing with cross-entropy, KL-divergence measures the absolute difference. When $$p(x)=q(x)$$, $$H(p||q)=0$$. 


## logistic regression
sigmoid function $$f(x; w)=\frac{1}{1+e^{-wx}}$$: convert linear classification result to a probability.

maximize likelihood => solve the parameters in linear classification. 

The key here is that sigmoid function normalize raw result into 0 and 1. Then we can construct below likelihood.

{% math %}
\max P(w|Y) <=> \max P(Y|X, w) = \Pi_{i=1}^n f^{y_i}(x_i; w)(1-f(x_i;w))^{1-y_i}
{% endmath %}

Apply logarithm on both side.
{% math %}
\log P(y|x, w) = \sum_{i=1}^n y_i\log f(x_i;w) + (1-y_i)\log (1-f(x_i; w))
{% endmath %}

This is equivalent to cross-entropy. And then we can apply gradient descend to optimization.

Check [here](https://tech.meituan.com/intro_to_logistic_regression.html) for details.



## Support Vector Machine and Core function
Will do


## Kalman filter
It is a algorithm for accurately estimating or predicting based on multiple observed data.

A classic example is SLAM in which we have multiple data collecting sensors like odometry, IMU and visual features. The Kalman filter is to solve how to reliably combine all sensor data and estimate a accurate pose.


## Cross-entropy instead of mean square error (MSE) as the loss function in classification?
when we do classification, we usually apply softmax (this is an important premise) to normalize the output value to 0-1. 
In this case, the gradient of MSE loss will be prone to 0 when the prediction is closed to 0 or 1. This will lead to slow convergence.
On the contrary, the gradient of cross-entropy is linear with the prediction changing. When the prediction is closed to the label, the gradient will be small and vice versa.

### * Cross-entropy
{% math %}
p(x_i) = softmax(x_i) = \frac{e^{x_i}}{\sum_i e^{x_i}} \\
f(x) = -\sum_i y_i \log p(x_i)
{% endmath %}
where $$f(x)$$ is the objective cross-entropy function. To minimize it, we calculate the gradient w.r.t. $$x_i$$

{% math %} 
\frac{\partial f(x)}{\partial x_i} = - \frac{y_i}{p(x_i)}p^{'}(x_i) \\ 
p^{'}(x_i) = \frac{\partial p(x_i)}{\partial x_i} = p(x_i) - p^2(x_i)
{% endmath %}
So we have 
{% math %}
\frac{\partial f(x)}{\partial x_i} = - \frac{y_i}{p(x_i)}p^{'}(x_i) = -y_i (1-p(x_i))
{% endmath %}
When $$p(x_i)$$ is closed to 1, the gradient will be prone to 0. (Remember that we use one-hot encoding to represent $$\mathbf{y}, \mathbf{x}$$)


### * MSE
{% math %}
f(x) = \sum_i (y_i - p(x_i))^2 \\ 
\frac{\partial f(x)}{\partial x_i} = 2(y_i - p(x_i))(-p^{'}(x_i)) = -2(y_i - p(x_i))(p(x_i) - p^2(x_i))
{% endmath %}
So we have the gradient 
{% math %}
\frac{\partial f(x)}{\partial x_i} = -2p(x_i)(1-p(x_i))(y_i - p(x_i))
{% endmath %}
When $$p(x_i)$$ is closed to 0 and 1, the gradient will be closed to 0. This is not desirable and will slow down the learning process. Because if $$y_i = 1, p(x_i) = 0$$, we hope the learning step is large, but the gradient is 0.




## The property of softmax
1. property: the big value will take a large portion of the probability.
2. potential drawback: exponential problem.


## Covariance and correlation coefficients
Covariance is a measure of how two variables change together, but its magnitude is unbounded, so it is difficult to interpret. By dividing covariance by the product of the two standard deviations, one can calculate the normalized version of the statistic. This is the correlation coefficient.
1. Covariance: represent the unnormalized correlation.
2. Correlation coefficient: represent the normalized correlation.
{% math %}
\rho_{xy} = \frac{Cov(x, y)}{\sigma_x \sigma_y}
{% endmath %}

Refer to [here](https://www.investopedia.com/terms/c/correlationcoefficient.asp) for details.


## Normal distribution and multivariate normal distribution
+ One-variable
{% math %}
f(x) = \frac{1}{\sqrt{2\pi} \sigma} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
{% endmath%}

+ Multi-variables
{% math %}
f(X=x_1,...,x_n) = \frac{1}{\sqrt{(2\pi)^n |C(X)|}} e^{-\frac{1}{2}(X-\bar X)^T C^{-1}(X)(X-\bar X)}
{% endmath %}
where $$|C(X)|$$ is the determinant of covariance matrix C(X). By the way, C(X) is usually denoted as $$\mathbf{\Sigma}$$ in many literatures.

{% math %}
X = \left[x_1, x_2, ..., x_n \right]^T \\

\bar X = mean \\

C(X) = 
\begin{pmatrix}
cov(x_1, x_1) & cov(x_1, x_2) & ... & cov(x_{n-1}, x_{n}) \\
 & ... & &  \\
cov(x_n, x_1) & cov(x_n, x_2) & ... & cov(x_{n}, x_{n})  \\
\end{pmatrix} 
{% endmath %}

Refer to [here](https://en.wikipedia.org/wiki/Multivariate_normal_distribution) for details.