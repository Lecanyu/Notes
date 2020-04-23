---
layout: post
title: Probabilistic Stuffs
---

## How to simulate a random number which satisfy a probabilistic distribution 
1. Inverse transform method
2. Acceptance rejection method

Check [here](http://blog.codinglabs.org/articles/methods-for-generating-random-number-distributions.html)


## Confidence Interval and variance
In statistics, confidence interval is a type of estimate computed from the statistics of the observed data. 
It gives a range of values for an unknown parameter (e.g. the mean). 
The interval has an associated confidence level that the true parameter (e.g. the mean) is in the proposed range.

Variance is the expectation of the squared deviation of a random variable from its mean. 
Informally, it measures how far a set of (random) numbers are spread out from their average value




## How to shuffle an array
Given an array $$a = [1,2,3,...,n]$$, design an algorithm to evenly and randomly shuffle it.
There are two algorithms. Which one is correct?

{% highlight cpp %} 
for i=1 to n do swap(a[i], a[random(1,n)]);
for i=1 to n do swap(a[i], a[random(i,n)]);
{% endhighlight %}

The second one is correct.

The second algorithm is that you randomly select a number from $$i$$ to $$n$$ and put that number on the $$i$$-th position.
Obviously, there are totally $$n!$$ possible combinations.

In contrast, the first algorithm will generate total $$n^n$$ combinations. Since $$\frac{n^n}{n!}$$ is not a integer number, some combinations are more likely appeared.


## How to calculate probability density function (概率密度函数)  
{% sidenote 1, 'Put a common conclusion here. If $$X \sim N(\mu_1, \sigma_1^2), Y \sim N(\mu_2, \sigma_2^2)$$, then $$Z=X+Y$$ will satisfy $$Z \sim N(\mu_1+\mu_2, \sigma_1^2+\sigma_2^2)$$. The derivation is [here](https://en.wikipedia.org/wiki/Sum_of_normally_distributed_random_variables).'%}
Given a random variable $$X$$ and its probability density function $$f(X)$$, if another random variable $$Y=X^2$$, what is the probability density function of $$Y$$?

A common mistake is that you put $$X = \pm \sqrt Y$$ into $$f(X)$$, and then calculate $$f(\sqrt Y)$$ (when $$X>0$$), and $$f(-\sqrt Y)$$ (when $$X<0$$).

This is correct when $$f(X)$$ is a standard function mapping (常规的函数映射).

However, this is wrong in probability density function.

The probability density function represent how possible a random variable drop in a interval $$[-\infty, x]$$.
{% math %}
P(-\infty \le X \le x) = \int_{-\infty}^x f(X) dX = F(X=x) - F(X=-\infty)
{% endmath %}
where $$F(X)$$ is the primitive function (原函数).

To calculate the probability density function of $$f(Y)$$, you should go from its primitive function.
{% math %}
F_Y(y) = P(Y \le y) = P(X^2 \le y ) = P(-\sqrt y \le X \le \sqrt y) = \int_{- \sqrt y}^{\sqrt y} f(X) dX = F_X(\sqrt y) - F_X(-\sqrt y)
{% endmath %}


The derivation in Chinese is listed below.
{% maincolumn 'assets/machine_learning/probability_dense_function_calculation.png'%}

<span style="color:red"> 需要搞清楚一种通用且普适的方法用于计算新随机变量的概率密度函数. 一种可能的做法或许可以参考[here](https://www.zhihu.com/question/37400689)</span>
