# Machine Learning (Fall 2011) - Homework 1

### By: Daniel Foreman-Mackey (<danfm@nyu.edu>)

The source code for this project is available at: <https://github.com/dfm/ml-class>.
While this file should be readable as plain text, it can be seen rendered properly
at: <https://github.com/dfm/ml-class/blob/master/README.md>.

I implemented all the requirements in Python instead of LUSH since it is what I 
feel comfortable with and what I use daily in my own research. The code requires
[NumPy](http://numpy.scipy.org/) to be installed on the system and it also needs 
[SciPy](http://scipy.org/) (but only for the direct solution to the linear system).
The main algorithms are implemented in the `linear` module (in the source file
`linear/linear.py`) and all the user facing code is in `hw1.py`. The `database`
module in `database/database.py` reads in the dataset and should be easily 
extensible to other machine learning datasets.

You can run

    python hw1.py --help

to see a list of command line arguments and an explaination of the code usage.

## Implementation

The `Perceptron`, `LinearRegression` and `LogisticRegression` algorithms are
implemented in `linear/linear.py` as subclasses of the abstract `LinearClassifier`
class. The `LinearClassifier` does the heavy lifting for training and testing of the 
algorithms and the subclasses provide their own `loss_function` and `delta` methods.
Not surprisingly, `loss_function` returns the value of the loss function given a 
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

## Hyperparameters

For logistic regression, the optimal value of `eta` seems to be between 0.002 
and 0.005. For `eta` > 0.005, the algorithm quickly diverges and for `eta` < 0.002,
the algorithm converges very slowly. The three plots below show (as a function 
of `eta`):

1. the total number of iterations before convergence,
2. the loss calculated on the training set and
3. the fractional error calculated on the training set.

To generate the data for these plots, run

    python hw1.py --logistic --hyperparams --ntrain 1000

![](https://github.com/dfm/ml-class/raw/master/hyperparams/Niter.png)

![](https://github.com/dfm/ml-class/raw/master/hyperparams/loss.png)

![](https://github.com/dfm/ml-class/raw/master/hyperparams/ferr.png)

The same experiment can be run on linear regression by running

    python hw1.py --linear --hyperparams --ntrain 1000

This experiment shows that linear regression for classification is very unstable.
For the algorithm to converge at all, `eta` must be \lesssim 10^-4 but the actual
value for convergence seems to be very sensitive to the particular training sample.
Logistic regression, however, seems to be quite stable in the range listed above.

## Convergence

The stopping criterion that I used to detect convergence is just a simple threshold
on the relative difference in the training loss between the current and previous
iteration. It would probably be even better to implement a decaying learning rate
in the future.

## Training set size

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

To run this test, run

    python hw1.py --size

## Regularization

I implemented L1 and L2 regularization in the `LinearClassifier` base class.
Therefore, both `LinearRegression` and `LogisticRegression` can be modified to
include regularization by constructing the object with the keyword argument
`alpha` or `beta`:

    machine = LogisticRegression(data, eta=0.002, alpha=0.2)

or 

    machine = LogisticRegression(data, eta=0.002, beta=0.1)

For convenience, I defined the regularization terms as `0.5*alpha*||W||^2` and
`0.5*beta*sum(|W_i|)`.  Therefore, the direct solution of the linear regression
system (with L2 regularization) is

    W = (alpha*I + X^T X)^-1 X^T t

instead of

    W = (X^T X)^-1 X^T t

### Optimal _alpha_

To improve the performance on small datasets, I ran a grid in `alpha` and `beta`
for 10, 30 and 100 training samples and the results are shown in the following 3
figures. For `N=10`, the optimal `alpha` is ~0.8; for `N=30`, it is ~0.25; and for
`N=100`, it is ~0.15.

![](https://github.com/dfm/ml-class/raw/master/alpha10.png)

![](https://github.com/dfm/ml-class/raw/master/alpha30.png)

![](https://github.com/dfm/ml-class/raw/master/alpha100.png)

### Optimal _beta_

The results for `beta` are shown in the following three figures. This L1
regularization doesn't seem to improve the efficiency of `LogisticRegression` for
`N=10` but for `N=30` and `N=100`, `beta`~0.27 and ~0.07 (respectively) provide
the optimal results.

![](https://github.com/dfm/ml-class/raw/master/beta10.png)

![](https://github.com/dfm/ml-class/raw/master/beta30.png)

![](https://github.com/dfm/ml-class/raw/master/beta100.png)

