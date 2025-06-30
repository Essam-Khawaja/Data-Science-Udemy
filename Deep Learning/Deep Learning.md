# Deep Learning

## Neural Networks
Creating a machine learning algorithm means getting the right output from an input. Before getting comfortable about the output, we must train the model properly so that it can work with new data confidently.

To train the model:
1. Data: We need a lot of data with the correct outputs.
2. Model: Find a model (such as a linear one) to fit the data
3. Objective Function: Estimates how correct the models outputs are on average
4. Optimization Algorithm: Vary the parameters to get better objective functions

Types of Machine Learning:
1. Supervised: We provide the algorithm with a set of inputs and the correct corresponding outputs. It then learns to produce outputs close to the ones we are looking for.
2. Unsupervised: We feed inputs with no target outputs. We just ask the model to find some pattern or underlying logic behind the data.
3. Reinforcement: We train a model based on the rewards it receives. Basically, the algorithm does a bunch of stuff based on what it receives good feedback for.

Supervised Learning can be split into classification and regression. For this section of the course, we will only be doing supervised learning.

### The Linear Model
Lets suppose we have two variables x and y, and we wish to find the linear model between them. We have pairs of data where, 
$$y = f(x)$$

Now, in order to find the model, we give the algorithm a bunch of these pairs so that it can find the pattern needed between x and y in terms of coefficients. 

$$f(x) = xw + b$$

In this case, w is the weight of the input x, and b is the bias. 

Note that we can have more than one input. Instead of adding multiple variables and weights, we simply make x a row vector, and the w a column vector. Then, when calculating y, we simply multiply the vectors. Pretty sure this can be done with matrices as well when having more than one output.

### Objective Function
The measure used to evaluate how well the model's outputs match the desired correct values. There are two types: 

1. Loss Functions: The lower the loss function, the higher the accuracy of the model.
2. Reward Functions: The higher the reward function, the higher the accuracy. Usually used in reinforcement.

The target (T) is the expected output from the algorithm.

There are two common loss functions. For regression, we use L2-Norm. It is the sum of squared errors between the outputs obtained and the corresponding targets. 

For classification, we use cross-entropy. This is really hard to write down, use lecture 328 for reference. 

### Optimization Algorithms
The most fundamental algorithm is the gradient descent. 
1. Find the derivative of the function f(x) as f'(x)
2. Set a random x0 (such as x0 = 4)
3. Then, iteratively calculate the formula:
    $$x_{i+1} = x_i - \eta f'(x_i)$$

    where $$\eta (eta)$$ is the learning rate.

4. Iterate through until f'(xi) becomes 0 (the value of x becomes constant), meaning that we get:
$$x_{i+1} = x_i$$

Generally, we want the learning rate to be high enough where we can reach the closest minimum in a rational amount of time, but low enough such that we don't vary around the minimum arbitrarily.

