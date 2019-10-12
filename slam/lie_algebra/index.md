---
layout: post
title: Lie Group and Lie Algebra
---

Here I try to briefly answer several important but intutive questions about Lie group and algebra.
{%sidenote 1 'Other explanation is [here](https://blog.csdn.net/weicao1990/article/details/83375148).'%}

For the math, please check [here](https://blog.csdn.net/heyijia0327/article/details/50446140) and slambook in chapter 4.

## What are Lie group and Lie algebra
A set of number and one calculation operation can compose a *group*, as long as this kind of calculation satisfies 4 conditions (封闭性、结合律、幺元、有逆).

In SLAM, the rotation matrix and transformation matrix satisfy those conditions, and thus, they are also called 旋转群 SO(3) and 欧式群 SE(3).

Every Lie group has its Lie algebra, which describes the local property of Lie group. 

On symbols, we usually have

Lie group SO(3) <-> Lie algebra so(3)

Lie group SE(3) <-> Lie algebra se(3)


## Why we need Lie group and Lie algebra
The main motivation is for calculating derivatives w.r.t. matrix. 

Since rotation and transformation don't have add calculation (rotation1 add rotation2 won't be a rotation anymore), the standard derivative calculation rule cannot be applied directly. You should consider the special properties of rotation and transformation as extra constraints in calculation which brings many limits and inconveniences.

In contrast, Lie algebra can be represented as vector which has add operation, and the standard derivative calculation rule can be applied.
{%sidenote 2 'The add operation on Lie algebra is equivalent to the multiply on Lie group with a Jacobian term (i.e. BCH approximation, BCH近似). Check the slam book for the detail. '%}

## How to calculate derivatives w.r.t. pose in Lie algebra
1. The rotation or transformation matrix can be seen as a Lie group.

2. Find its Lie algebra by Logarithmic mapping.
   <br/> (i.e. The original matrix is represented by Lie algebra in which the variable is vector).

3. Calculate the derivatives w.r.t. those vector variable. 
   <br/> There are two approaches: based on standard derivative rule and based on perturbation model. 
   <br/> The standard derivative rule generates result with BCH approximation jacobian term, and perturbation model can give a more tidy result which is usually used.


## The conversion between Lie group and Lie algebra
{% maincolumn 'assets/slam/Lie_group_and_Lie_algebra.png'%}
