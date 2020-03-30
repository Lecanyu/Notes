---
layout: post
title: Overview (Interview)
---

为方便今后回顾，这里我总结一些笔试和面试经常考的问题，包括通用的数学、算法、编程语言、特定方向如机器学习、计算机视觉方面的知识。


## 数学优化和概率
牛顿迭代法

概率密度函数的相关计算，洗牌算法（shuffle的实现）
>见前面的[Probabilistic stuffs](../../machine_learning/probabilistic_stuffs)


## 算法
以下算法题都是我曾在笔试或者面试中遇见的，应该涵盖了经常考察的知识点，题目如果文字描述过于简单可以在网上简单搜一下，下面的多数题目我都曾实现过（有时间我将建个github 把每道题的描述和解法放上去，最好分门别类的进行整理）


下一个全排列

输出所有组合数

排列组合中的卡特兰数
> 递推公式类似这种a(n)=a(0)×a(n-1)+a(1)×a(n-2)+...+a(n-1)×a(0) https://www.cnblogs.com/wuyuegb2312/p/3016878.html


约瑟夫环问题

A* 算法
> 启发式函数 f = g + h


单调栈求直方图中的最大矩形面积
> 可拓展到01数组中最大的全1子矩阵

最长回文子串
> 动态规划O(n^2)，效率更高的Manacher算法...

包含一个字符串中K个字母的最短子串


求N个数中的前K个最大值
> 堆排序，c++中的优先队列 priority_queue

堆排序
> 建堆O(n)，调整堆O(logn)

手写快速排序


给定平面中的n个点，求一条能过最多点的直线
> 思路：对直线的k,b建立hash_map，然后遍历C_n^2次，共O(n^2)


判断是否是有向无环图（拓扑排序，实质BFS）

最早完成时间，最晚完成时间、关键路径
> 一道题涵盖了有向无环图、关键路径这两个问题：https://blog.csdn.net/qq_41376345/article/details/80483501

最长公共子序列（子序列连续和不连续的情况）

最长递增的子数组（子数组连续和不连续的情况）

最小生成树 （Kruskal算法，贪心和并查集）

图中的最短路径（Dijkstra算法，广度优先搜索，更新距离数组、是否访问过的数组）

用移位操作实现加法（不用+ - * / 四则运算）

数的原码、反码、补码。（为什么用补码表示负数 -- 一套加法算法可以同时处理正数和负数）

给定树的前序和中序遍历序列，重建整颗树（递归重建）

两个栈模拟双端队列

最长的递增子数组 
> 两种算法：普通dp O(n^2) 和带trick的dp O(nlogn) https://blog.csdn.net/u013178472/article/details/54926531 )

字典序排序

字典序中第K个数字

连续子数组的最大和

和为S的连续正数序列

递增序列中，和为S的两个数字

平衡二叉树 *

旋转数组，找最小值，能否用递归做

任意四边形的IOU 
> 多边形剪裁算法+shoelace公式

数组中和为S的两个数字

两个递增数组中第K大的数 O(logK)

所有子数组最大值之和

0/1背包 *




## 编程语言和设计模式

面向对象的特性（封装、继承、多态）

C++的静态多态和动态（运行时）多态


C++中的虚函数表和虚函数指针
{% sidenote 1, "详细的介绍见[这里](https://www.learncpp.com/cpp-tutorial/125-the-virtual-table/) "%}
> 原理很简单，这里简单说一下，类中一旦出现了虚函数，那么编译器在编译时则会为该类添加一个虚函数表，用于索引对应的虚函数。
同时还会为该类添加一个隐藏的虚函数指针成员变量，该指针指向虚函数表的地址。
运行时多态主要是体现在当把子类对象（实例）指针或引用转化为父类时发生的，在这个情况下，虚函数指针指向的依旧是子类的虚函数表，因此调用虚函数时依旧是运行子类的函数。
其他情况下，运行的则是父类的函数。
建议看我在VS2017中OOP project中写的例子。


C++11 STL中的智能指针（shared_ptr, unique_ptr, weak_ptr）和move的语义 {% sidenote 1, "详细的介绍见[这里](https://www.geeksforgeeks.org/auto_ptr-unique_ptr-shared_ptr-weak_ptr-2/) "%}


C++中的左值和右值的概念 {% sidenote 1, "详细的介绍见[这里](https://www.ibm.com/developerworks/cn/aix/library/1307_lisl_c11/index.html) "%}


Python中的传值和传引用 
> 不可变对象都是传值，可变对象都是传引用

python中is和==的区别
> Python中对象包含的三个基本要素，分别是：id(身份标识)、type(数据类型)和value(值)。对象之间比较是否相等可以用==，也可以用is。
is和==都是对对象进行比较判断作用的，但对对象比较判断的内容并不相同。
is比较的是两个对象的id值是否相等，也就是比较两个对象是否为同一个实例对象，是否指向同一个内存地址。
==比较的是两个对象的内容是否相等，默认会调用对象的__eq__()方法。

常见的设计模式
>单例模式, 工厂模式, 代理模式, 适配器模式, 观察者模式

单例模式
>其意图是保证一个类仅有一个实例，并提供一个访问它的全局访问点，该实例被所有程序模块共享

C++中构造单例
> 1.私有化它的构造函数，以防止外界创建单例类的对象；2.使用类的私有静态指针变量指向类的唯一实例；3.使用一个公有的静态方法获取该实例。
需要注意多线程的竞争。建议看我在VS2017中OOP project中写的例子。



## 机器学习

Precision和Recall
>Recall=TP/TP+FN，衡量放走了多少正例；Precision=TP/TP+FP，衡量把多少负例拿进来了


TPR和FPR，AUC，ROC
>ROC曲线y轴：实际正样本中分对的比率TPR=TP/TP+FN，x轴：实际负样本中分错的比率FPR=FP/FP+TN。
AUC(Area under curve)是曲线所围的面积

K-means和KNN 
>K-means 无监督,需训练. KNN有监督,无需训练过程

高维向量的查询
>KD-tree和基于KNN-Graph的NN-Descent算法

特征降维
>PCA（无监督）和LDA(有监督的数据降维)

神经网络的宽度和深度
>宽度通常指feature的depth，深度指layer的叠加

卷积核filter size(width, height, in_depth, out_depth)和feature map的维度计算
>width_next_layer = floor[(width +2 * pad - filter_w)/stride] + 1

logistic regression
>损失函数的意义:极大似然估计

SVM的损失函数推导
>普通的SVM的目标函数 min 1/2 W^T* W  s.t. y(Wx+b)>=1、核函数、软间隔

Bagging和boosting的区别和联系
>见前面的[misc](../../machine_learning/miscellaneous)

embedding的作用是什么 
>数据降维，用转化成某种特征向量，使其向量之间的距离能表达相似性，具体应用：word2vec和metric learning

神经网络中的梯度消失和梯度膨胀是什么，怎么解决
>梯度剪裁,Wasserstein GAN用了这种技术。BN。使用合理的activation function

激活函数的作用
>引入非线性因素，比如sigmoid、ReLU、Leaky ReLU

如何选出好特征，去掉不好的特征
>大多数情况下是根据经验和实验的方法。比较通用的经验方法：根据特征的方差。实验方法：先随机选一些特征，然后训练，看效果

如何检验避免过拟合
>dropout，data augmentation，加正则，减少参数，提前停止训练

偏差和方差问题
>通常而言，如果出了问题，弱模型的bias大（欠拟合），强模型variance大（过拟合）

Siamese net, triplet net and their loss function

dropout和BN训练和测试的差异
>dropout仅在训练时需要，测试时不需要，在训练时还要对它的输出进行放大以保证其数值大小与测试时一致。
BN训练用的是min-batch的统计量，测试时用的moving_average

Group normalization
>见前面的[misc](../../machine_learning/miscellaneous)

depthwise卷积和pointwise卷积
>主要是为了模型压缩 https://zhuanlan.zhihu.com/p/80041030

CNN是否有平移和旋转不变性
>由于CNN不好的可解释性，这个问题暂时存在争议。不过一般认为滑动卷积始终可以找到feature，并且pooling操作可以一定程度上带来不变性

SGD使用mini batch优化和使用所有样本优化各有什么优缺点
>mini batch：适应内存，存在跳出局部最优的可能性，主要缺点是收敛速度略慢。
所有样本优化：收敛快，但通常数据很多，内存不允许这样做




## 计算机视觉


### 2D图像视觉

Faster RCNN中的RPN(Region proposal network)结构

FPN(Feature pyramid network)结构

OHEM (Online hard example mining)
>所有proposal都参与向前传播，计算loss后，选择loss较大且IoU满足阈值要求的框进行向后传播计算导数，更新参数

Faster RCNN、YOLO、SSD
>one stage和two stage, anchor和anchor free。
Faster RCNN是two stage有anchor。
YOLO是one stage无anchor。
SSD是one stage有anchor。

NMS, Soft NMS, Softer NMS



### 3D视觉和SLAM

简单描述特征点ORB,SIFT等
>ORB - 速度快，通常用在实时的SLAM中、旋转、平移不变。
SIFT - 速度比ORB慢，旋转、平移、尺度不变

Bundle adjustment

Pose graph

Optical flow

Structure from motion

Fundamental matrix, Essential matrix, Homography matrix
>本质矩阵5自由度，3旋转，3平移，减去scale。
基础矩阵7自由度，9参数 - scale - 秩为2即行列式=0。
单应矩阵8自由度，9参数 - scale。


稀疏性和边缘化


李群和李代数


卡尔曼滤波
>运动和观测方程，隐马尔可夫模型，极大似然估计和最大后验估计


GN,LM优化方法
>见前面的[misc](../../slam/optimization)



## 简历项目和论文

Learning Continually from Low-shot Data Stream
    
    -
    motivation: Achieve incremental learning on low shot data

    contribution: 
        0. A new interesting problem
        1. dynamically balance learning object and regularization term.  
        2. A multi-step training algorithm to learn intrinsic feature (maximize gradient inner product among different data points) from low shot data  
    
    algorithm: 
        1. convert to qudratic programming with constraint
        2. A multi-step algorithm like OpenAI's Reptile

    Points:
        MAS, Path Integral, EWC and other CL methods like GEM



JigsawNet: Shredded Image Reassembly using Convolutional Neural Network and Loop-based Composition
    
    -
    motivation: Reassemble image/document fragments (Just like solve irregular jigsaw puzzles)

    pipeline:
        pairwise matching -> CNN filtering -> global reassembing (NP-hard, like SAT problem (SAT is NP-complete) in a sense)

    contribution: 
        1. Design a new CNN to recognize how good a pairwise matching is
            - new designed CNN with shallow feature stacks and RoIAlign attention mechanism
            - Oversampling and downsampling
            - boosting algorithm with CNN to solve imbalanced within-class problem

        2. Develop a loop-based algorithm to solve global composition.
            - instead of greedy algorithm, we apply loop closure on pose graph to find more reliable solution.
            - bottom-top merge



Sparse3D: A new global model for matching sparse RGB-D dataset with small inter-frame overlap
    
    -
    motivation: Reconstruct indoor scenes from sparse RGB-D data (small inter-frame overlap)

    contribution:
        1. multiple feature ensemble (2D feature: SIFT, Shi-tomasi. 3D feature: NARF)
            - We use graph matching rather than RANSAC. (Construct the affinity matrix to encode the spacial coherency among feature points.)

        2. global pruning and optimization model to maximize mutual consistency of alignments
            - initialize using loop closure
            - optimize pose graph with elastic term


Fast Global Registration ( a more robust registration method)
    
    -
    features:
        - Give dense correspondences, no initialization.
        - Automatically disable false matching. (Unlike ICP, this algorithm no correspondence update, and closest point queries)



## HR相关的问题
最大性格缺点：
>正向的说法：
不满足于现状，
对自己的水平永远不满足，总是主动去学习新的东西。
反向的说法：
对于不变的，死板的工作和环境非常容易失去兴趣，
在安定的环境里做一成不变的工作，并且无法提出改善的时候会让我失去工作的动力，更适合充满挑战，变化的环境。


如何问公司情况？
>链接：https://www.zhihu.com/question/21941315/answer/666506337


