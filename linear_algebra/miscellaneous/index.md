---
layout: post
title: Miscellaneous (linear algebra)
---

## Singular value decomposition (SVD)
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


## Principle component analysis (PCA)
{% math %}
A = V \Sigma V^T
{% endmath %}

where $$A$$ usually is a covariance matrix which is a symmetrical matrix. $$V$$ is composed by a series of eigen vectors which are mutually orthonormal. $$\Sigma$$ is corresponding eigen value.



## Covariance matrix























