# GradientDescent
This is an implementation of Gradient Descent algorithm frequently encountered in Machine Learning/Neural Networks. 
This implementation is heavily influenced by [QuantitativeBytes implementation](https://www.youtube.com/watch?v=BjkmFVv4ccw), 
I have enhanced the original by using C++17 lambdas instead of function pointers and replacing finite difference derivatives with Automatic Algorithmic Differentiation using AutoDiff 

Gradient Descent is a numerical technique for finding the minimum of a function.
For example, consider function <img src="https://bit.ly/1HR0GAY" align="center" border="0" alt="f(x)=x^2" width="78" height="22" /> as shown below. 
We know that analytically the first derivative is <img src="https://bit.ly/35c8FqD" align="center" border="0" alt=" f'(x) = 2x" width="87" height="21" />

![alt text](https://github.com/suhasghorp/GradientDescent/raw/master/src/fx2.png "f(x)=x^2")
