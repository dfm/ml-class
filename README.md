# Machine Learning (Fall 2011) - Homework 1

### By: Daniel Foreman-Mackey (<danfm@nyu.edu>)

## Implementation

The Perceptron, LinearRegression and LogisticRegression algorithms are implemented
in `linear/linear.py` as subclasses of the abstract LinearClassifier class.  The
LinearClassifier does the heavy lifting for training and testing of the algorithms
and the subclasses provide their own `loss_function` and `delta` methods. Not
surprisingly, `loss_function` returns the value of the loss function given a 
particular label and x vector.  `delta` return the gradient of `loss_function`,
__divided by x__ since the x vector factors out of the update term in stochastic
gradient descent in the same way for each of these algorithms.  The LinearRegression
object has one extra method `solve` that uses the direct solution to solve the 
system.

The code to read in the dataset is in `dataset/dataset.py` and it includes all the
required helper functions (e.g. `normalize`, `shuffle`, etc.).  This module also
includes a `LinearlySeperableDataset` that generates a truly linearly seperable
mock dataset to run tests on.  To test the perceptron algorithm (for example) on
this dataset, run:

    python hw1.py --test --perceptron

## Experiments with _spambase_

To test the algorithms on the spambase dataset with 1000 training samples, you 
can just run

    python hw1.py --perceptron --linear --direct --logistic --ntrain 1000

If you add the `--verbose` option, it will output the test and training error and
loss at each iteration of the training phase (_warning_: this is _very_ slow).

### Hyperparameters

For logistic regression, the optimal value of \eta seems to be between 0.002 
and 0.005. For \eta > 0.005, the algorithm quickly diverges and for \eta < 0.002,
the algorithm converges very slowly. The three plots below show (as a function 
of \eta):
1. the total number of iterations before convergence,
2. the loss calculated on the training set and
3. the fractional error calculated on the training set.

![](https://github.com/dfm/ml-class/raw/master/hyperparams/Niter.png)

![](https://github.com/dfm/ml-class/raw/master/hyperparams/loss.png)

![](https://github.com/dfm/ml-class/raw/master/hyperparams/ferr.png)

Questions
---------

2 - Experiments with the Spambase dataset

      - set the training set size to 1000 
        and the test set size to 1000.
      - The stochastic gradient method for linear regression
        and logistic regression require you to find good 
        values for the learning rate (the step size, eta).
    a - what learning rates will cause linear regression
        and logistic regression to diverge?
    b - what learning rate for linear regression 
        and logistic regression that produce
 	the fastest convergence?
    c - implement a stopping criterion that 
        detects convergence.

    d - train logistic regression with 10, 30, 100, 500, 
        and 3000 training samples, and 1000 test samples. 
        for each size of training set, provide:
        - the final value of the average loss, and 
	  the classification errors on the training 
	  set and the test set
        - the value of the learning rate used
        - the number of iterations performed

    e - what is the asymptotic value of the training/test 
        error for very large training sets?

3 - L2 and L1 regularization
    When the training set size is small, it is often helpful
    to add a regularization term to the loss function.
    The most popular ones are:
    L2 norm:   alpha*||W||^2    (aka "ridge regression")
    L1 norm:   beta*[ SUM_i |W_i| ]  (aka "LASSO")

    a - how is the linear regression with direct solution
        modified by the addition of an L2 regularizer.
    b - Modify you logistic regression code to add the
        L2 and L1 regularizers to the cost function.
	Can you improve the performance on the test set
	for training set sizes of 10, 30 and 100.
	What value of alpha nad beta give the best results?


