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

知乎中的回答提供了一种借助狄拉克函数来计算新随机变量的概率密度函数的通用计算方法。
{% sidenote 1, '狄拉克函数的介绍和性质 [Dirac delta function](https://www.wikiwand.com/zh-cn/%E7%8B%84%E6%8B%89%E5%85%8B%CE%B4%E5%87%BD%E6%95%B0#/%E7%B8%AE%E6%94%BE%E8%88%87%E5%B0%8D%E7%A8%B1).'%}

具体表述如下：
随机变量$$X$$符合某概率分布$$P_X(x)$$，对$$X$$进行某种变换后得到一个新的随机变量$$Z$$，即$$z = f(x)$$，那么$$Z$$对应的概率密度函数$$P_Z(z)$$可以如下计算
{% math %}
P_Z(z) = \int_{-\infty}^{\infty} P_X(x)\delta(z-f(x)) dx
{% endmath %}
这里$$\delta(z-f(x))$$是狄拉克函数，仅在$$\delta(0)$$处非0，且$$\int_{-\infty}^{\infty} \delta(x) dx = 1 $$，上面积分表达式的含义是
随机变量$$Z=z$$处的概率密度函数由所有满足$$z=f(x)$$的$$X$$概率密度函数求和得到，这是符合概率直觉的。

这里的关键的就是怎么计算带狄拉克函数的积分，下面以上面的$$Y=X^2$$为例进行说明，关于多元变量的概率密度函数的计算例子可以参考[here](https://www.zhihu.com/question/37400689)。
根据前面介绍的，我们有
{% math %}
f_Y(y) = \int_{-\infty}^{\infty} f_X(x)\delta(y-x^2) dx
{% endmath %}
显然，当$$x=\pm \sqrt y$$时，$$y-x^2=0$$。
由于$$\delta(y-x^2)$$只在$$x_0 = \pm \sqrt y$$时不为0，我们可以将$$f_X(x)$$在积分运算中当作常数，也即$$f_Y(y) = f_X(x_0) \int_{-\infty}^{\infty} \delta(y-x^2) dx$$。
令$$ t = y - x^2 $$，那么$$ \frac{dt}{dx}=-2x $$，所以$$dx_0 = \frac{dt}{-2x_0}$$
{% math %}
f_Y(y) = f_X(x_0) \int_{-\infty}^{\infty} \frac{\delta(t)}{-2x_0} dt = f_X(x_0) \int_{-\infty}^{\infty} \frac{\delta(t)}{|2x_0|} dt
{% endmath %}
这里第二个等号把$$-2x_0$$改成了$$|2x_0|$$，
原因在于狄拉克函数积分的几何性质，即坐标轴进行放缩$$a$$，为了使积分面积依然是1，其高度也要相应进行$$\frac{1}{|a|}$$倍的放缩，具体请阅读上面旁注中的链接。

很显然，由于$$x_0 = \pm \sqrt y$$
{% math %}
f_X(x_0) \int_{-\infty}^{\infty} \frac{\delta(t)}{|2x_0|} dt = \frac{1}{|2\sqrt y|}f_X(\sqrt y) + \frac{1}{|-2\sqrt y|}f_X(-\sqrt y) = \frac{1}{2\sqrt y}[f_X(\sqrt y) + f_X(-\sqrt y)]
{% endmath %}
这与上面的答案是相同的。

