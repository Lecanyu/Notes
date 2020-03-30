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

Recall越低，很多正例被错误分类（放走了很多正例）
Precision越低，很多负例被错误分类（包进了很多负例）


## ROC and AUC 
{% maincolumn 'assets/machine_learning/AUC_ROC.png'%}
{% maincolumn 'assets/machine_learning/ROC_curve.png'%}
ROC is the curve based on TPR and FPR.

AUC is the area under the ROC curve.

AUC作为评价指标，可以有效处理样本不平衡问题。比如：在垃圾邮件分类中，垃圾邮件的label = 1，只占总样本的1%；非垃圾邮件的label = 0，占总样本99%。
如果一个naive分类器把所有的样本分成0，那么也有99%的精确度。
但是如果用AUC作为评价指标时，可以得到TPR=0（实际是垃圾邮件的，分对了多少），FPR=0（实际非垃圾邮件，分错了多少），不管分类的阈值怎么取，始终都是坐标系中的点(0, 0)，因此AUC=0 ?
<span style="color:red"> (需要进一步搞清楚AUC的计算公式)</span>


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

## Feature dimensionality reduction
There two common methods: PCA and LDA.
PCA is an unsupervised dimensionality reduction method, whereas LDA is supervised.

The main difference between them:

PCA try to extract new feature vector which has big variance.
LDA try to project original space to low dimensional space so as to maximize the distance between different classes and minimize the within-class variance.
{% sidenote 1, "Please check [here](https://www.cnblogs.com/pinard/p/6244265.html) for details about LDA. "%}


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
If we use axes= [0,1,2] to calculate (e.g. mean, var = tf.nn.moments(input, axes=[0, 1, 2])), then the output of mean, var will be a 1-D vector with size=depth. 
If we use axes=[0] to calculate, then the output of mean, var will be a 3-D vector with size=[height, width, depth]. The picture below demonstrates this 3-D vector case.

{% maincolumn 'assets/machine_learning/batch_normalization_ex.png'%}


## Group normalization
The comparison between several normalization methods
{% maincolumn 'assets/machine_learning/group_normalization.png'%}

Batch normalization has several defects:

1. BN only works fine with large batch size, but this cannot guaranteed on complex tasks (e.g. object detection) due to insufficient memory.
2. BN use mini-batch's mean and variance to normalize during training, but use moving-average (滑动平均) mean and variance to normalize during testing. It introduces inconsistency, especially when data distribution between training and testing are different.

Unlike BN to normalize on dimension [batch, width, height], 
the group normalization (GN) try to normalize on dimension [width, height, channel=k] (k is a hyperparameter). It doesn't matter with batch. So it solves the first defect.
Moreover, GN always use group mean and variance during training and testing. {% sidenote 10, "IN, LN and GN doesn't use batch dimension to normalize."%}



## Recurrent Neural Network (RNN)
An tutorial and introduction about [RNN](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/)

{% maincolumn 'assets/machine_learning/RNN.png'%}

## Hidden Markov Model 
An specific dice [example](https://www.zhihu.com/question/20962240) will help to understand.

The basic elements:
Transition matrix, emission matrix, initial state.

Three basic problems:
1. Given the HMM model, how to calculate the probability of an observation. (forward algorithm)
2. Given the HMM model and a sequence of observations, how to estimate which the most possible hidden state sequences are. (Viterbi algorithm, dynamic programming)
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
Since $$P(\theta|D)=\frac{P(D|\theta)P(\theta)}{P(D)}$$, if we can assume the prior is a constant, then $$ \max P(\theta|D) = \max P(D|\theta)$$.
Maximum likelihood estimation is widely used in solving various machine learning models, especially when prior is unknown.


2. Maximum A Posterior (MAP).
Sometimes, we may not be able to make such assumption that the prior is a constant. Instead, prior may satisfy a distribution. In this case, the target is  $$P(\theta|D) =  \mathop{\arg\min}_{\theta} P(D|\theta)P(\theta)$$. Since every data is generated independently, we have $$ P(\theta|D) =  \mathop{\arg\min}_{\theta} \Pi_i^n P(D=x_i|\theta)P(\theta) $$.
If we use prior $$P(\theta)$$ to represent the complexity of a model, then MAP is linked with structure risk minimization. 


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

Generally, we also have below property without graph structure if {%m%}A, B{%em%} are independent.
{%math%} P(A, B|C) = P(A|C)P(B|C) {%endmath%}
Because {%m%}P(A, B, C) = P(A, B|C)P(C) = P(A|B, C)P(B|C)P(C) = P(A|C)P(B|C)P(C){%em%}

## Several concepts need to be distinguished
1. Bayesian Network
2. Markov Random Field
3. Conditional Random Field
4. Markov Chain
5. Hidden Markov Model
6. Markov Decision Process

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


## Logistic regression
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



## Support Vector Machine
Please check [here1](https://blog.csdn.net/u014433413/article/details/78427574) and [here2](https://zhuanlan.zhihu.com/p/31652569) for the details about naive SVM, SVM with kernel and SVM with soft margin.

Here, we give a brief introducation.

In SVM, we want to find a hyperplane $$wx+b = 0$$ (w, b are the parameters to learn), so that this hyperplane can correctly divide two classes (the labels are -1, 1) data points.
{% maincolumn 'assets/machine_learning/SVM1.png'%}
We define that the data points which are the nearest ones to hyperplane $$wx+b=0$$ should locate on $$wx+b=1$$ (for positive datapoint) and $$ wx+b=-1 $$ (for negative datapoint). The reason why this definition makes sense is that we can always adjust the learned hyperplane by multiple a factor (e.g. k) without change the hyperplane in space 
(i.e. $$wx+b=0$$ <=> $$k(wx+b)=0$$).

So our goal is to maximize the distance between $$wx+b=1$$ and $$wx+b=-1$$. Obviously, the distance is 
$$\frac{2}{||w||}$$. We can write the objective function as below
{% math %}
\min \frac{1}{2} w^T w \\
s.t. \sum_i y_i(wx_i+b) \ge 1
{% endmath %}
To solve this objective, we apply Lagrangian multipler to convert original constrained problem to 
{% math %}
\max_{\lambda} \min_w \frac{1}{2} w^T w - \sum_i \lambda_i [y_i(wx_i+b)-1] \\
s.t. \lambda \ge 0
{% endmath %}
Then calculate the partial gradients w.r.t. $$w, b$$ first, then $$\lambda$$. {% sidenote 2, "This is the naive SVM" %}

Sometimes, the datasets cannot be divided by linear plane. So we usually use kernel tricks, which introduce nonlinear property in classifier. 
The main idea of kernel trick is to find a mapping $$\phi(x)$$ to map original space to a higher space. The picture below gives an example. {% sidenote 3, "This is the SVM with kernel. Check [here](https://www.zhihu.com/question/24627666) for the reasons why kernel can map to a higher dimensions. " %}
{% maincolumn 'assets/machine_learning/SVM_kernal.png'%}
The P mapping in above picture is the $$\phi(x)$$. You should distinguish the kernel function and dimensional mapping function.

However, sometimes, kernel can still fail. So people introduce the soft margin, which allows misclassification on some data points.
The objective is {% sidenote 4, "This is soft margin." %}
{% math %}
\min \frac{1}{2} w^T w + c\sum_i \xi_i \\
s.t. \sum_i y_i(wx_i+b) \ge 1 - \xi_i \\
\xi_i \ge 0
{% endmath %}
The $$\xi_i$$ can be seen as hinge loss: $$\xi_i = \max (0, 1-y_i(wx_i+b))$$. And SVM can be explained in hinge loss.
{% math %}
\min \frac{1}{m}\sum_i^m \max (0, 1-y_i(wx_i+b)) + k||w||^2
{% endmath %}



## Kalman filter
It is a algorithm for accurately estimating or predicting based on multiple observed data. 
See [here](https://www.zhihu.com/question/22422121) for an intuitive explanation and example.

A classic example is SLAM in which we have multiple data collecting sensors like odometry, IMU and visual features. The Kalman filter is to solve how to reliably combine all sensor data and estimate a accurate pose.

**In linear case, the Kalman filter problem can be formulated as**
{% math %}
x_k = A_k x_{k-1} + u_k + w_k\\
z_k = C_k x_{k} + v_k
{% endmath %}
The first equation is motion equation where $$x_k$$ is the state at $$k$$-th moment. $$u_k$$ is motion measurement with noise $$w_k$$ which is satisfied a gauss distribution $$w_k \sim N(0, R_k)$$. 

The second equation is observation equation where $$z_k$$ is the observation measurement with noise $$v_k$$ which is also satisfied a gauss distribution $$v_k \sim N(0, Q_k)$$.

*Since the motion and observation measurements are noisy (subject to gauss distribution) which means that the current state estimation is noisy too, our goal now is to find an optimal state estimation which has the minimum uncertainty (i.e. minimum covariance).*

The way is to calculate Kalman Gain which is the optimal weight between motion and observation. {% sidenote 1, "I ignore the derivation of Kalman Gain, since it is a little bit complicated. You can search online about this."%}

The complete algorithm is:

First, predict currect state from previous state based on motion equation. 
(The previous state can be represented by gauss distribution $$N(\hat x_{k-1}, \hat P_{k-1})$$)

{% math %}
\bar x_k = A_k \hat x_{k-1} + u_k \\
\bar P_{k} = A_k \hat P_{k-1} A_{k}^T + R 
{% endmath %}
This equation is easy to understand according to the properties of gauss distribution (The new mean and new covariance from two gauss distribution). 

Second, calculate the Kalman Gain
{% math %}
K = \bar P_k C_k^T (C_k \bar P_k C_k^T + Q_k)^{-1}
{% endmath %}

Finally, correct/modify the rough estimation from motion equation.
{% math %}
\hat x_k = \bar x_k + K (z_k - C_k \bar x_k) \\ 
\hat P_k = (I - KC_k) \bar P_k
{% endmath %}

If you derivate a little, you can find that if $$Q_k = 0$$ (no covariance in observation), then $$\hat x_k = C_k^{-1} z_k$$, which means the final estimation is totally depended on observation equation.  There is a similar conclusion when $$R_k = 0$$.

Now we know the idea of Kalman filter in linear system.

In reality, the system is usually nonlinear (e.g. SLAM problem). To apply above idea, we need to extend KF to nonlinear case. The Taylor expansion can do this.




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


## Some important concepts in machine learning (statistic learning)
**1. Hypothesis space**

Hypothesis space is a set of models which map the input to output.
When hypothesis space is given, the range of learning space is determined.
For example, if we plan to use a neural network as the model, the hypothesis space consists of all possible parameters of this neural network.


**2. Expected risk, Empirical risk, Structure risk**

In machine learning, we usually suppose all data are i.i.d (independently drawn from identical distribution).
The expected risk is defined as 
{% math %}
R_{exp}(f) = E(L(Y, f(X))) = \int_{X, Y} L(y, f(x))P(x, y)dxdy
{% endmath %}
where $$(X,Y)$$ are training data. $$L(\cdot, \cdot)$$ is loss function. $$P(X, Y)$$ is joint probability distribution.
The ideal learning procedure is to find a best $$f$$ which can minimize $$R_{exp}(f)$$.
However, it is impossible to know joint probability distribution $$P(X, Y)$$ beforehand.
Because there is no need to learn if we already know $$P(X, Y)$$. 
{% sidenote 1, "Therefore, machine learning is usually an ill-posed problem. About the ill-posed problem, you can check [here](https://en.wikipedia.org/wiki/Well-posed_problem) and [here](https://stats.stackexchange.com/questions/433692/why-is-pattern-recognition-often-defined-as-an-ill-posed-problem)"%}

In practice, we minimize empirical risk instead of expected risk.
{% math %}
R_{emp}(f) = \frac{1}{N} \sum_{i=1}^{N} L(y_i, f(x_i))
{% endmath %}
Obviously, $$R_{exp}(f)$$ is the loss w.r.t. joint probability distribution which is absolutely accurate.
$$R_{emp}(f)$$ is the loss w.r.t. the average of training data which is an approximation.
According to law of large numbers, this approximation could be accurate when giving enough training data.

Structure risk is similar with empirical risk. The only difference is that it introduces regularization term to penalize complex model to relieve overfitting.
{% math %}
R_{str}(f) = \frac{1}{N} \sum_{i=1}^{N} L(y_i, f(x_i)) + \lambda J(f)
{% endmath %}


**3. Cross validation, k-fold cross validation**

Cross validation is a strategy to select a model with best generalization ability.
When we have enough labeled data, we can simply split them into training, validation, testing sets and use validation set to select model. 
K-fold cross validation is also a common strategy when data are relatively insufficient.

The general procedure is as follows:
1. Shuffle the dataset randomly.
2. Split the dataset into k groups
3. For each unique group:
    1. Take the group as a hold out or test data set
    2. Take the remaining groups as a training data set
    3. Fit a model on the training set and evaluate it on the test set
    4. Retain the evaluation score and model
4. Select a model with best score


**4. Generative model, Discriminative model**

If a model try to learn joint probability distribution $$P(X, Y)$$, it is a generative model.
If a model learn conditional probability distribution $$P(Y|X)$$ (i.e. decision boundary), it is a discriminative model.

These two concepts here are different from they are in GAN (generative adversarial network).


**5. Learning Bias (Inductive Bias)**

The introduction from [wiki](https://en.wikipedia.org/wiki/Inductive_bias) 

In machine learning, models or algorithms are trained on training dataset and make predictions on testing datasets which are unseen data.
Without any additional assumptions, this problem cannot be solved exactly since unseen situations might have an arbitrary output value. 
The kind of necessary assumptions about the nature of the target function are subsumed in the phrase inductive bias



## The difference between bagging and boosting, the advantage of assembly learning

The difference see [here](https://www.cnblogs.com/liuwu265/p/4690486.html).

I make a brief summary here.

Bagging (usually used in random forest)
1. Bootstrap Sampling from training data (random sampling without replacement). 
2. All samples with same weights in training. 
3. K seperate models with same importance. So all models can be trained parallelly

Boosting (e.g. adaptive boosting method)
1. Training on all data with weights (misclassified data will be more important in next training round)
2. K seperate models with different importance. 


Why Bagging improve performance?

There are two explanations:
1. It can reduce the variance of single decision tree since model may fit some minor features if we use all training data. Minor feature fitting leads to high variance (over-fitting). 
2. From probabilistic perspective, the probability of wrong prediction from multiple models is low. 


Why Boosting improve performance?

The goal of optimization is to minimize the misclassified data. 

