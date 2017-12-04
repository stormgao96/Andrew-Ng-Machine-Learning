# Andrew Ng - Machine Learning



[TOC]





## Week 1

### Ⅰ. Introduction

####1. What is Machine Learning?

Two definitions of Machine Learning are offered. 

- Arthur Samuel described it as: "the field of study that gives computers the ability to learn without being explicitly programmed." This is an older, informal definition.
- Tom Mitchell provides a more modern definition: "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."



Example: playing checkers.

E = the experience of playing many games of checkers

T = the task of playing checkers.

P = the probability that the program will win the next game.

In general, any machine learning problem can be assigned to one of two broad classifications:
Supervised learning and Unsupervised learning.



#### 2. Supervised Learning

In supervised learning, we are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and the output.

``在监督学习中，我们给定一个数据集并且已经知道了正确的输出应该是什么样的，知道在输入和输出间存在某种关系``

Supervised learning problems are categorized into "regression" and "classification" problems. In a regression problem, we are trying to predict results within a continuous output, meaning that we are trying to map input variables to some continuous function. In a classification problem, we are instead trying to predict results in a discrete output. In other words, we are trying to map input variables into discrete categories.

``监督学习问题被分为回归问题和分类问题。在回归问题中，我们尝试预测连续输出的结果，这意味着我们正试图将输入变量映射到一些连续函数。在分类问题中，我们试图预测离散输出的结果，换句话说，我们正试图将输入变量映射到离散类别。``

Example 1:Given data about the size of houses on the real estate market, try to predict their price. Price as a function of size is a continuous output, so this is a regression problem.We could turn this example into a classification problem by instead making our output about whether the house "sells for more or less than the asking price." Here we are classifying the houses based on price into two discrete categories.

Example 2:(a) Regression - Given a picture of a person, we have to predict their age on the basis of the given picture

(b) Classification - Given a patient with a tumor, we have to predict whether the tumor is malignant or benign.



#### 3. Unsupervised Learning 

Unsupervised learning allows us to approach problems with little or no idea what our results should look like. We can derive structure from data where we don't necessarily know the effect of the variables.

``无监督学习使我们能够在不知道结果是什么样的情况下解决问题。我们可以从数据中推导出结构，而不一定需要知道变量会造成的影响。`` 

We can derive this structure by clustering the data based on relationships among the variables in the data.

``我们可以通过基于数据中变量之间的关系对数据进行聚类来推导出这种结构。``

With unsupervised learning there is no feedback based on the prediction results.

``在无监督学习的基础上，没有基于预测结果的反馈。``

Example:

Clustering: Take a collection of 1,000,000 different genes, and find a way to automatically group these genes into groups that are somehow similar or related by different variables, such as lifespan, location, roles, and so on.

Non-clustering: The "Cocktail Party Algorithm", allows you to find structure in a chaotic environment. (i.e. identifying individual voices and music from a mesh of sounds at a cocktail party).



### Ⅱ. Model and Cost Function

#### 1. Model Representation

$m$ = Number of training examples 样本个数

$x^{(i)}$ = "input" variables / features 输入的变量（特征）

$y^{(i)}$ = "output" variable / "target" variable 输出的变量，目标变量

$(x^{(i)}, y^{(i)})$ = a training example 训练样本

$(i)$ = an index into the training set, has nothing to do with exponentiation 在训练集中的索引，不是做幂指数运算



Also

$(i)$ = an index into the training set, has nothing to do with exponentiation 在训练集中的索引，不是做幂指数运算

$X$ = the space of input values

$Y$ = the space of output values

$X$ = $Y$ = $R$ 



To describe the supervised learning problem slightly more formally, our goal is, given a training set, to learn a function h : X → Y so that h(x) is a “good” predictor for the corresponding value of y. For historical reasons, this **function h is called a hypothesis**. Seen pictorially, the process is therefore like this:

![img](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/H6qTdZmYEeaagxL7xdFKxA_2f0f671110e8f7446bb2b5b2f75a8874_Screenshot-2016-10-23-20.14.58.png?expiry=1512086400000&hmac=j1_ppSzPy_FcSo6obPy6Tl5M1zPU8k6NhZgOj-f0V6U)

When the target variable that we’re trying to predict is continuous, such as in our housing example, we call the learning problem a regression problem. When y can take on only a small number of discrete values (such as if, given the living area, we wanted to predict if a dwelling is a house or an apartment, say), we call it a classification problem.



#### 2. Cost Function

We can measure the accuracy of our hypothesis function by using a **cost function**. This takes an average difference (actually a fancier version of an average) of all the results of the hypothesis with inputs from x's and the actual output y's.

$J(\theta_0, \theta_1) = \cfrac{1}{2m}\sum\limits_{i=1}^{m}(\hat y_i - y_i)^2 = \cfrac{1}{2m}\sum\limits_{i=1}^{m}(h_\theta(x_i) - y_i)^2$

To break it apart, it is  $\cfrac{1}{2}\overline{x}$ where $\overline{x}$ is the mean of the squares of $h_\theta(x_i) - y_i$ , or the difference between the predicted value and the actual value.

This function is otherwise called the "Squared error function", or "Mean squared error". The mean is halved $(\cfrac{1}{2})$as a convenience for the computation of the gradient descent, as the derivative term of the square function will cancel out the $\cfrac{1}{2}$ term. The following image summarizes what the cost function does:



#### 3. Cost Function - Intuition I

If we try to think of it in visual terms, our training data set is scattered on the x-y plane. We are trying to make a straight line (defined by hθ(x)) which passes through these scattered data points.

`如果我们试图用视觉的方式来思考它，训练数据集就散布在x-y平面上。我们试着去做一条直线（由h定义）通过这些散布的数据点。`

Our objective is to get the best possible line. The best possible line will be such so that the average squared vertical distances of the scattered points from the line will be the least. Ideally, the line should pass through all the points of our training data set. In such a case, the value of J(θ0,θ1) will be 0. The following example shows the ideal situation where we have a cost function of 0.

`我们的目标是获得最佳线路。尽可能好的点是这样的，从线的散射点的平均垂直距离将是最小的。理想情况下，这条线应该通过我们训练数据集中的所有点。这种情况下，J的值将会是0。下面的例子展示了代价函数值为0的理想情况。`

![img](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/_B8TJZtREea33w76dwnDIg_3e3d4433e32478f8df446d0b6da26c27_Screenshot-2016-10-26-00.57.56.png?expiry=1512086400000&hmac=kfSaaSyD_mfkyTJ-6_JS76KZ8TfbyeBhuV2A3eMnhhI)

When θ1=1, we get a slope of 1 which goes through every single data point in our model. Conversely, when θ1=0.5, we see the vertical distance from our fit to the data points increase.

`当θ1=1时，我们得到了一条斜率为1的直线，经过了模型中的每一个数据点。相反，当θ1=0.5时，可以看到拟合到数据点的垂直距离增加` 

![img](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/8guexptSEeanbxIMvDC87g_3d86874dfd37b8e3c53c9f6cfa94676c_Screenshot-2016-10-26-01.03.07.png?expiry=1512086400000&hmac=tjwLbg1CISBMfcnXACFUvXvb8bNnwZfxymCSVI_TVDM)

This increases our cost function to 0.58. Plotting several other points yields to the following graph:

`这使我们的代价函数值增长到0.58。绘制其他几点到下面的图表`

![img](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/fph0S5tTEeajtg5TyD0vYA_9b28bdfeb34b2d4914d0b64903735cf1_Screenshot-2016-10-26-01.09.05.png?expiry=1512086400000&hmac=cue4ZZ2oaoNPwqJx6BYJ3Fm1g-eVrkz4xoWqnoWeUhQ)

Thus as a goal, we should try to minimize the cost function. In this case, θ1=1 is our global minimum.

`因此，作为一个目标，我们应该尽量减少代价函数的值。在这种情况下，θ1=1是我们的全局最小值`



#### 4. Cost Function - Intuition II

A contour plot is a graph that contains many contour lines. A contour line of a two variable function has a constant value at all points of the same line. An example of such a graph is the one to the right below.

![img](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/N2oKYp2wEeaVChLw2Vaaug_d4d1c5b1c90578b32a6672e3b7e4b3a4_Screenshot-2016-10-29-01.14.37.png?expiry=1512086400000&hmac=4PrVWXbYZuoeVj_8CgNYKHVzLMJleGBiR42znZzFAh8)

Taking any color and going along the 'circle', one would expect to get the same value of the cost function. For example, the three green points found on the green line above have the same value for J(θ0,θ1) and as a result, they are found along the same line. The circled x displays the value of the cost function for the graph on the left when θ0 = 800 and θ1= -0.15. Taking another h(x) and plotting its contour plot, one gets the following graphs:

![img](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/26RZhJ34EeaiZBL80Yza_A_0f38a99c8ceb8aa5b90a5f12136fdf43_Screenshot-2016-10-29-01.14.57.png?expiry=1512086400000&hmac=Px4W4Is-Zr52QhjIOrbBYI3lFyCOloqFUsK97DfvIDA)

When θ0 = 360 and θ1 = 0, the value of J(θ0,θ1) in the contour plot gets closer to the center thus reducing the cost function error. Now giving our hypothesis function a slightly positive slope results in a better fit of the data.

![img](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/hsGgT536Eeai9RKvXdDYag_2a61803b5f4f86d4290b6e878befc44f_Screenshot-2016-10-29-09.59.41.png?expiry=1512086400000&hmac=jwfl6zL0wEatCNCBT1eX4VUWS4z0oK7ngdrYP0G1NXk)

The graph above minimizes the cost function as much as possible and consequently, the result of θ1 and θ0 tend to be around 0.12 and 250 respectively. Plotting those values on our graph to the right seems to put our point in the center of the inner most 'circle'.


### Ⅲ. Parameter Learning
#### 1. Gradient Descent

So we have our hypothesis function and we have a way of measuring how well it fits into the data. Now we need to estimate the parameters in the hypothesis function. That's where gradient descent comes in.

Imagine that we graph our hypothesis function based on its fields θ0 and θ1 (actually we are graphing the cost function as a function of the parameter estimates). We are not graphing x and y itself, but the parameter range of our hypothesis function and the cost resulting from selecting a particular set of parameters.

We put θ0 on the x axis and θ1 on the y axis, with the cost function on the vertical z axis. The points on our graph will be the result of the cost function using our hypothesis with those specific theta parameters. The graph below depicts such a setup.

![img](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/bn9SyaDIEeav5QpTGIv-Pg_0d06dca3d225f3de8b5a4a7e92254153_Screenshot-2016-11-01-23.48.26.png?expiry=1512086400000&hmac=nsQi5fPQ-Kwb2PuxlRU4iV6kbeEWKvpJbgFT_lqkD3o)

We will know that we have succeeded when our cost function is at the very bottom of the pits in our graph, i.e. when its value is the minimum. The red arrows show the minimum points in the graph.

The way we do this is by taking the derivative (the tangential line to a function) of our cost function. The slope of the tangent is the derivative at that point and it will give us a direction to move towards. We make steps down the cost function in the direction with the steepest descent. The size of each step is determined by the parameter α, which is called the learning rate.

For example, the distance between each 'star' in the graph above represents a step determined by our parameter α. A smaller α would result in a smaller step and a larger α results in a larger step. The direction in which the step is taken is determined by the partial derivative of J(θ0,θ1). Depending on where one starts on the graph, one could end up at different points. The image above shows us two different starting points that end up in two different places.

The gradient descent algorithm is:

repeat until convergence:

$\theta_j := \theta_j - \alpha\cfrac{\partial}{\partial\theta_j}J(\theta_0, \theta_1)$

where

j=0,1 represents the feature index number.

At each iteration j, one should simultaneously update the parameters θ1,θ2,...,θn. Updating a specific parameter prior to calculating another one on the j(th) iteration would yield to a wrong implementation.

![img](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/yr-D1aDMEeai9RKvXdDYag_627e5ab52d5ff941c0fcc741c2b162a0_Screenshot-2016-11-02-00.19.56.png?expiry=1512086400000&hmac=Bl8-4iQCfSldMIokYIsemWddmBBV669XKlNbyz7sxqo)





#### 2. Gradient Descent Intuition

In this video we explored the scenario where we used one parameter $\theta_1$ and plotted its cost function to implement a gradient descent. Our formula for a single parameter was :

Repeat until convergence:

$\theta := \theta_1 - \alpha\cfrac{d}{d\theta_1}J(\theta_1)$

Regardless of the slope's sign for $\cfrac{d}{d\theta_1}J(\theta_1)$, $\theta_1$eventually converges to its minimum value. The following graph shows that when the slope is negative, the value of θ1 increases and when it is positive, the value of θ1 decreases.

![img](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/SMSIxKGUEeav5QpTGIv-Pg_ad3404010579ac16068105cfdc8e950a_Screenshot-2016-11-03-00.05.06.png?expiry=1512086400000&hmac=1tOExzee9bgNEoLpzKCM-ruvYrWeY0-CAvNWqGSR1g4)

On a side note, we should adjust our parameter α to ensure that the gradient descent algorithm converges in a reasonable time. Failure to converge or too much time to obtain the minimum value imply that our step size is wrong.

![img](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/UJpiD6GWEeai9RKvXdDYag_3c3ad6625a2a4ec8456f421a2f4daf2e_Screenshot-2016-11-03-00.05.27.png?expiry=1512086400000&hmac=a8mjbVrufsSs0g-AZrk_R-Pt9AIkqBQ2Zo4ireQluGg)

How does gradient descent converge with a fixed step size α?

The intuition behind the convergence is that $\cfrac{d}{d\theta_1}J(\theta_1)$ approaches 0 as we approach the bottom of our convex function. At the minimum, the derivative will always be 0 and thus we get:

$\theta_1 := \theta_1 - \alpha * 0$

![img](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/RDcJ-KGXEeaVChLw2Vaaug_cb782d34d272321e88f202940c36afe9_Screenshot-2016-11-03-00.06.00.png?expiry=1512086400000&hmac=3Rp-iT__guUsOh6GmsB62EgXz03p21Wva4AdfOUxbMA)



#### 3. Gradient Descent For Linear Regression

When specifically applied to the case of linear regression, a new form of the gradient descent equation can be derived. We can substitute our actual cost function and our actual hypothesis function and modify the equation to :

repeat until convergence:{

​					$\theta_0=\theta_0 - \alpha\cfrac{1}{m}\sum\limits_{i=1}^{m}(h_\theta(x_i) - y_i)$

​					$\theta_1 = \theta_1 - \alpha\cfrac{1}{m}\sum\limits_{i=1}^m((h_\theta(x_i) - y_i)x_i)$

}

where m is the size of the training set, θ0 a constant that will be changing simultaneously with θ1 and xi,yiare values of the given training set (data).

Note that we have separated out the two cases for θj into separate equations for θ0 and θ1; and that for θ1 we are multiplying xi at the end due to the derivative. The following is a derivation of ∂∂θjJ(θ) for a single example :

![img](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/QFpooaaaEea7TQ6MHcgMPA_cc3c276df7991b1072b2afb142a78da1_Screenshot-2016-11-09-08.30.54.png?expiry=1512086400000&hmac=lFfrYeiCVxla_mmf9ghzWp8cWRhUccm1InhR_NhfZFk)

The point of all this is that if we start with a guess for our hypothesis and then repeatedly apply these gradient descent equations, our hypothesis will become more and more accurate.

So, this is simply gradient descent on the original cost function J. This method looks at every example in the entire training set on every step, and is called **batch gradient descent**. Note that, while gradient descent can be susceptible to local minima in general, the optimization problem we have posed here for linear regression has only one global, and no other local, optima; thus gradient descent always converges (assuming the learning rate α is not too large) to the global minimum. Indeed, J is a convex quadratic function. Here is an example of gradient descent as it is run to minimize a quadratic function.

![img](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/xAQBlqaaEeawbAp5ByfpEg_24e9420f16fdd758ccb7097788f879e7_Screenshot-2016-11-09-08.36.49.png?expiry=1512086400000&hmac=iDxS54jQY79k_iDUjuoxNMvZcmiLwmbqtOG5-S6yGlU)

The ellipses shown above are the contours of a quadratic function. Also shown is the trajectory taken by gradient descent, which was initialized at (48,30). The x’s in the figure (joined by straight lines) mark the successive values of θ that gradient descent went through as it converged to its minimum.



### Ⅳ. Linear Algebra Review

#### 1. Matrices and Vectors

Matrices are 2-dimensional arrays:
$$
\begin{bmatrix}
   a & b & c \\
   d & e & f \\
   g & h & i \\
   j & k & l \\
  \end{bmatrix}
$$
The above matrix has four rows and three columns, so it is a 4 x 3 matrix.

A vector is a matrix with one column and many rows:

$\begin{bmatrix}w \\ x \\ y \\ z \\\end{bmatrix}$

So vectors are a subset of matrices. The above vector is a 4 x 1 matrix.



**Notation and terms**:

- $A_{ij}$ refers to the element in the $i^{th}$ row and $j^{th}$ column of matrix A.
- A vector with 'n' rows is referred to as an 'n'-dimensional vector.
- $v_i$ refers to the element in the $i^{th}$ row of the vector.
- In general, all our vectors and matrices will be 1-indexed. Note that for some programming languages, the arrays are 0-indexed.
- Matrices are usually denoted by uppercase names while vectors are lowercase.
- "Scalar" means that an object is a single value, not a vector or matrix.
- $R$ refers to the set of scalar real numbers.
- $R^n$ refers to the set of n-dimensional vectors of real numbers.

Run the cell below to get familiar with the commands in Octave/Matlab. Feel free to create matrices and vectors and try out different things.

```` 
% The ; denotes we are going back to a new row.
A = [1, 2, 3; 4, 5, 6; 7, 8, 9; 10, 11, 12]

% Initialize a vector 
v = [1;2;3] 

% Get the dimension of the matrix A where m = rows and n = columns
[m,n] = size(A)

% You could also store it this way
dim_A = size(A)

% Get the dimension of the vector v 
dim_v = size(v)

% Now let's index into the 2nd row 3rd column of matrix A
A_23 = A(2,3)
````



#### 2. Addition and Scalar Multiplication

Addition and subtraction are **element-wise**, so you simply add or subtract each corresponding element:

$\begin{bmatrix}a & b \\ c & d \\\end{bmatrix} + \begin{bmatrix}w & x \\ y & z \\\end{bmatrix} = \begin{bmatrix}a+w & b+x \\ c+y & d+z \\\end{bmatrix}$

Subtracting Matrices:

$\begin{bmatrix}a & b \\ c & d \\\end{bmatrix} - \begin{bmatrix}w & x \\ y & z \\\end{bmatrix} = \begin{bmatrix}a-w & b-x \\ c-y & d-z \\\end{bmatrix}$

To add or subtract two matrices, their dimensions must be **the same**.

In scalar multiplication, we simply multiply every element by the scalar value:

$\begin{bmatrix}a & b \\ c & d \\\end{bmatrix} * x = \begin{bmatrix}a*x & b*x \\ c*x & d*x \\\end{bmatrix}$

In scalar division, we simply divide every element by the scalar value:

$\begin{bmatrix}a & b \\ c & d \\\end{bmatrix} / x = \begin{bmatrix}a/x & b/x \\ c/x & d/x \\\end{bmatrix}$

Experiment below with the Octave/Matlab commands for matrix addition and scalar multiplication. Feel free to try out different commands. Try to write out your answers for each command before running the cell below.

```
% Initialize matrix A and B 
A = [1, 2, 4; 5, 3, 2]
B = [1, 3, 4; 1, 1, 1]

% Initialize constant s 
s = 2

% See how element-wise addition works
add_AB = A + B 

% See how element-wise subtraction works
sub_AB = A - B

% See how scalar multiplication works
mult_As = A * s

% Divide A by s
div_As = A / s

% What happens if we have a Matrix + scalar?
add_As = A + s
```



#### 3. Matrix-Vector Multiplication

We map the column of the vector onto each row of the matrix, multiplying each element and summing the result.

$\begin{bmatrix}a & b \\ c & d \\ e & f \\\end{bmatrix} * \begin{bmatrix}x \\ y\\\end{bmatrix} = \begin{bmatrix}a*x+b*y \\ c*x+d*y \\ e*x+f*y \\\end{bmatrix}$

The result is a **vector**. The number of **columns** of the matrix must equal the number of **rows** of the vector.

An **m x n matrix** multiplied by an **n x 1 vector** results in an **m x 1 vector**.

Below is an example of a matrix-vector multiplication. Make sure you understand how the multiplication works. Feel free to try different matrix-vector multiplications.

```
% Initialize matrix A 
A = [1, 2, 3; 4, 5, 6;7, 8, 9] 

% Initialize vector v 
v = [1; 1; 1] 

% Multiply A * v
Av = A * v
```



#### 4. Matrix-Matrix Multiplication

We multiply two matrices by breaking it into several vector multiplications and concatenating the result.

$\begin{bmatrix}a & b \\ c & d \\ e & f \\\end{bmatrix} * \begin{bmatrix}w & x \\ y & z \\\end{bmatrix} = \begin{bmatrix}a*w+b*y & a*x+b*z \\ c*w+d*y & c*x+d*z \\ e*w+f*y & e*x+f*z \\\end{bmatrix}$



An **m x n matrix** multiplied by an **n x o matrix** results in an **m x o** matrix. In the above example, a 3 x 2 matrix times a 2 x 2 matrix resulted in a 3 x 2 matrix.

To multiply two matrices, the number of **columns** of the first matrix must equal the number of **rows** of the second matrix.

For example:

```
% Initialize a 3 by 2 matrix 
A = [1, 2; 3, 4;5, 6]

% Initialize a 2 by 1 matrix 
B = [1; 2] 

% We expect a resulting matrix of (3 by 2)*(2 by 1) = (3 by 1) 
mult_AB = A*B

% Make sure you understand why we got that result
```



#### 5. Matrix Multiplication Properties

- Matrices are not commutative: A∗B≠B∗A
- Matrices are associative: (A∗B)∗C=A∗(B∗C)

The **identity matrix**(单位矩阵), when multiplied by any matrix of the same dimensions, results in the original matrix. It's just like multiplying numbers by 1. The identity matrix simply has 1's on the diagonal (upper left to lower right diagonal) and 0's elsewhere.

$\begin{bmatrix}1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \\\end{bmatrix}$

When multiplying the identity matrix after some matrix (A∗I), the square identity matrix's dimension should match the other matrix's **columns**. When multiplying the identity matrix before some other matrix (I∗A), the square identity matrix's dimension should match the other matrix's **rows**.

```
% Initialize random matrices A and B 
A = [1,2;4,5]
B = [1,1;0,2]

% Initialize a 2 by 2 identity matrix
I = eye(2)

% The above notation is the same as I = [1,0;0,1]

% What happens when we multiply I*A ? 
IA = I*A 

% How about A*I ? 
AI = A*I 

% Compute A*B 
AB = A*B 

% Is it equal to B*A? 
BA = B*A 

% Note that IA = AI but AB != BA
```



#### 6. Inverse and Transpose

The **inverse** of a matrix A is denoted A−1. Multiplying by the inverse results in the identity matrix.

A non square matrix does not have an inverse matrix. We can compute inverses of matrices in octave with the pinv(A) function and in Matlab with the inv(A) function. Matrices that don't have an inverse are *singular* or *degenerate*.

The **transposition** of a matrix is like rotating the matrix 90**°** in clockwise direction and then reversing it. We can compute transposition of matrices in matlab with the transpose(A) function or A':

$A=\begin{bmatrix}a & b \\ c & d \\ e & f \\\end{bmatrix}$

$A^T=\begin{bmatrix}a & c & e \\ b & d & f \\\end{bmatrix}$

In other words:

$A_{ij}=A_{ij}^{T}$

```
% Initialize matrix A 
A = [1,2,0;0,5,6;7,0,9]

% Transpose A 
A_trans = A' 

% Take the inverse of A 
A_inv = inv(A)

% What is A^(-1)*A? 
A_invA = inv(A)*A
```

