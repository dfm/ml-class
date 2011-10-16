Machine Learning (Fall 2011) - Homework 1
=========================================

Questions
---------

1 - implement:
    a - the perceptron learning algorithm
    b - linear regression with square loss 
        trained with stochastic gradient descent
    c - linear regression with square loss trained
        by direct solution of a linear system.
    d - the logistic regression algorithm trained 
        with stochastic gradient descent


    REMEMBER TO TAKE CARE OF THE BIAS PARAMETER!!!
    The bias is implemented a separate parameter,
    don't forget to update it in your code.

     ==> include your code as attachment.

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


