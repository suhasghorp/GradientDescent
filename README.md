# Gradient Descent
This is an implementation of Gradient Descent algorithm frequently encountered in Machine Learning/Neural Networks. 
This implementation is heavily influenced by [QuantitativeBytes implementation](https://www.youtube.com/watch?v=BjkmFVv4ccw), 
I have enhanced the original by using C++17 lambdas instead of function pointers and replacing finite difference derivatives with Automatic Algorithmic Differentiation using AutoDiff 

Gradient Descent is a numerical technique for finding the minimum of a function.
For example, consider function <img src="https://github.com/suhasghorp/GradientDescent/raw/master/images/1.png" align="center" border="0"> as shown below. 
We know that analytically the first derivative is <img src="https://github.com/suhasghorp/GradientDescent/raw/master/images/2.png" align="center" border="0">

<img src="https://github.com/suhasghorp/GradientDescent/raw/master/images/fx2.png" align="center" border="0">

## Gradient Descent Steps

Starting with point <img src="https://github.com/suhasghorp/GradientDescent/raw/master/images/4.png" align="center" border="0">, the equation to compute the next point <img src="https://github.com/suhasghorp/GradientDescent/raw/master/images/5.png" align="center" border="0"> is :

<img src="https://github.com/suhasghorp/GradientDescent/raw/master/images/3.png" align="center" border="0">

where <img src="https://github.com/suhasghorp/GradientDescent/raw/master/images/6.png" align="center" border="0"> is a constant, called Step Size or Learning Rate.





