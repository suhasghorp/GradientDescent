# GradientDescent
This is an implementation of Gradient Descent algorithm frequently encountered in Machine Learning/Neural Networks. 
This implementation is heavily influenced by [QuantitativeBytes implementation](https://www.youtube.com/watch?v=BjkmFVv4ccw), 
I have enhanced the original by using C++17 lambdas instead of function pointers and replacing finite difference derivatives with Automatic Algorithmic Differentiation using AutoDiff 

Gradient Descent is a numerical technique for finding the minimum of a function.
For example, consider function <img src="https://github.com/suhasghorp/GradientDescent/raw/master/images/1.png" align="center" border="0"> as shown below. 
We know that analytically the first derivative is <img src="https://github.com/suhasghorp/GradientDescent/raw/master/images/2.png" align="center" border="0">

<img src="https://github.com/suhasghorp/GradientDescent/raw/master/images/fx2.png" align="center" border="0">

## Gradient Descent Equation

<img src="https://github.com/suhasghorp/GradientDescent/raw/master/images/3.png" align="center" border="0">





