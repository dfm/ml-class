# Machine Learning (Fall 2011) - Homework 1

### By: Daniel Foreman-Mackey (<danfm@nyu.edu>)

The source code for this project is available at: <https://github.com/dfm/ml-class>.
While this file should be readable as plain text, it can be seen rendered properly
at: <https://github.com/dfm/ml-class/blob/master/README.md>.

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

### Convergence

The stopping criterion that I used to detect convergence is just a simple threshold
on the abosulte difference in the training loss between the current and previous
iteration. It would probably be even better to implement a decaying learning rate
in the future.

### Training set size

I trained the logistic regression classifier on 10, 30, 100, 500 and 3000 samples
and the results are listed in the following table. The asymptotic value of the
training/test error is ~9%.

                Training         Test
    Size  N   loss  %-error  loss  %-error
    ---- --- ------ ------- ------ -------
      10  36 0.3104   10.0  11.685   29.2
      30  17 1.7813   13.3  5.2616   26.7
     100  12 0.6504    7.0  4.0040   22.0
     500   6 0.7935   11.0  1.7380   12.7
    3000  26 0.5261    8.6  0.5429    9.7

Questions
---------

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


