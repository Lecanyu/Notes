---
layout: post
title: Miscellaneous (linear algebra)
---

## ***Singular value decomposition (SVD)***
{% math %}
A = U \Sigma V^T
{% endmath %}
where $$A$$ can be an arbitrary matrix (let’s say n x m). Then $$U$$ is a n x n matrix, $$\Sigma$$ is a n x m matrix, and $$V$$ is a m x m matrix. 

The column vectors in $$U$$ are mutually orthonormal, and the row vectors in $$V$$ are mutually orthonormal. 

If we pick only top-$$k$$ big singular value, then we have $$A \approx U_{n \times k} \Sigma_{k \times k} V_{k \times m}$$.

Here is a SVD example by using Eigen library.


{% highlight cpp %} 
#include <iostream>
#include <Eigen/Eigen>

using namespace Eigen;

int main()
{
    typedef Matrix<double, 2, 4> Matrix2x4;
    Matrix2x4 A;
    A<<1,1,0,2,
        -1,-1,0,-2;

    // A = U * S * V^T
    JacobiSVD<Matrix2x4> svd( A, ComputeThinU | ComputeThinV);      
    auto U = svd.matrixU();          // left matrix 2x2
    auto V = svd.matrixV();          // right matrix 4x4
    auto S = svd.singularValues();   // column vector, singular values
    MatrixXd S_mat = MatrixXd::Zero(U.cols(), V.cols());
    for(int i=0;i<std::min(S_mat.rows(), S_mat.cols()); ++i)
        S_mat(i,i) = S(i);

    std::cout<<"U:\n"<<U<<"\n";
    std::cout<<"sigma:\n"<< S_mat <<"\n";
    std::cout<<"V:\n"<<V<<"\n";
    std::cout<<"original matrix:\n"<< U * S_mat * V.transpose() <<"\n";

    return 0;
}
{% endhighlight %}


## ***Principle component analysis (PCA)***
{% math %}
A = V \Sigma V^T
{% endmath %}

where $$A$$ usually is a covariance matrix which is a symmetrical matrix. $$V$$ is composed by a series of eigen vectors which are mutually orthonormal. $$\Sigma$$ is corresponding eigen value.



## ***Covariance matrix***
Variance measure the concentration property within a series data
{% math %}
\frac{1}{n} \sum (x-\bar x)^2
{% endmath %}

Covariance measure the correlation property between two series data
{% math %}
\frac{1}{n} \sum (x-\bar x)(y-\bar y)
{% endmath %}

Covariance matrix measure the variance and covariance in n series/dimension data. For example, we have a sample of data as below

<!-- {% marginnote 'Table-ID4' 'Table 1: a simple statistic for demonstration.' %} -->

|          |**Height**|**Weight**|**Grade** |
|:--------:|:--------:|:--------:|
| Boy1     | 170cm    | 50kg     | 80       |
| Boy2     | ...      | ...      | ...      |
| Boy3     | ...      | ...      | ...      |


We can use a matrix $$A$$ to represent this data. 
Then the covariance matrix should be $$A^T A$$.


$$A^T A(0, 0), A^T A(1, 1), A^T A(2, 2)$$ is the variance in height, weight, grade dimension, respectively.

$$A^T A(0, 1)$$ is the covariance between height and weight dimensions. Other element in $$A^T A$$ can be interpreted in the same way.


## ***The solutions of linear equations***
Suppose we have linear equations:
{% math %}
Ax = b  \quad \textsf{(Non-homogeneous linear equations)}\\
Ax = 0 \quad \textsf{(Homogeneous linear equations)}
{% endmath %}
where $$A$$ is $$m$$ rows and $$n$$ cols.

We need to analyze the rank of matrix and its augmented matrix to figure out whether there is solution or not.

First, we can decompose $$Ax$$ to
{% math %}
Ax = \sum_i^n x_iA_{col_i}
{% endmath %}
It means the linear equations can be seen as linear combination of column vectors of matrix $$A$$.  


### * Non-homogeneous linear equations

It is intuitive to see that if $$R(A, b) > R(A)$$, $$b$$ is linearly independent with $$A$$.
It is impossible to find a vector combination to represent $$b$$. 
In other words, there is no solution. {% sidenote 1 '$$R(A)$$ means the rank of matrix A.'%}

When $$R(A, b) = R(A)$$, we can know $$b$$ is linearly dependent with $$A$$. 
So if $$R(A, b) = R(A) < n$$, $$Ax=b$$ has infinite solutions; 
if $$R(A, b) = R(A) = n $$, the equations has only one solution.

### * Homogeneous linear equations

Homogeneous case is similar with non-homogeneous. 
Since $$b=\boldsymbol{0}$$, if $$R(A) = n $$ (full ranked), there is only zero solution. 
If $$A$$ is not full ranked, we can find a linear combination of colunm vectors to represent the linear dependent ones. Therefore, there are infinite solutions.


## ***Matrix decomposition***
### * LU decomposition
When matrix $$A$$ is invertible, then we can decompose $$A = LU$$. We usually have two different composition: 
1. $$L$$ is lower unit triangular matrix (diagonal elements are 1) and $$U$$ is a upper triangular matrix; 
2. $$L$$ is lower triangular matrix and $$U$$ is an upper unit triangular matrix


### * Cholesky (LLT) decomposition
When matrix $$A$$ is a hermitian positive-definite matrix, then we can decompose $$A=LL^T$$, where $$L$$ is lower triangular matrix
{% sidenote 2, 'The main application of matrix decomposition is to solve Ax=b linear system. Decomposition is significant faster than directly calculating inverse matrix. '%}


### * Cholesky variant (LDLT) decomposition
It is the same with cholesky, but we don’t want to calculate the square root, which slow down the calculation and lose accuracy. 
So we can decompose $$A=LDL^T$$ instead, where $$L$$ is lower unit triangular matrix and $$D$$ is diagonal matrix

(LLT and LDLT has been introduced in [HERE](https://en.wikipedia.org/wiki/Cholesky_decomposition))


## ***Solving linear least squares systems***
There are three typical ways to solve linear least square system (SVD, QR factorization, Cholesky decomposition). 

Check [here](https://eigen.tuxfamily.org/dox/group__LeastSquares.html) for Eigen library code.

### * An example
I’d like to use a simple example to explain the Cholesky approach. 

Suppose we have 12 sample points on the 3D plane of equation $$z = 2x+3y$$ (with some noise). 
We can write these samples into a matrix.
{% math %}
A = 
\begin{pmatrix}
x_1 & y_1 \\
x_2 & y_2 \\
... & ... \\
x_{12} & y_{12}
\end{pmatrix} 
\quad
b = 
\begin{pmatrix}
2x_1+3y_1+noise_1 \\
2x_2+3y_2+noise_2 \\
...  \\
2x_{12}+3y_{12}+noise_{12}
\end{pmatrix} 
{% endmath %}

We want to calculate the coefficient $$ x=\begin{pmatrix} a\\ b \end{pmatrix} $$, 
such that
{% math %}
    \arg \min_x ||Ax-b||^2
{% endmath %}

If we calculate the partial derivation w.r.t. $$a,b$$, and let the partial derivations equal to 0.
We will have below linear equation.
{% math %}
    A^T Ax=A^T b
{% endmath %}
The solution of above equation is equivalent to the optimal estimation a, b. (it is easy to derive and prove).

To solve $$A^T Ax=A^Tb$$, we can apply Cholesky decomposition, but if $$A^TA$$ is ill-conditioned {%sidenote 2 'Check [here](https://en.wikipedia.org/wiki/Condition_number) for more details about ill-condition '%}
(e.g. $$A^TA$$ is not invertible), it will lead to problems.


## ***Pseudoinverse matrix***
Pseudoinverse matrix has two types:
1. Left pseudoinverse matrix.
2. Right pseudo inverse matrix.

For more details, check [HERE](https://www.qiujiawei.com/linear-algebra-16/). 





