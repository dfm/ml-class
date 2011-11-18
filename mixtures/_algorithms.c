#include <Python.h>
#include <numpy/arrayobject.h>

#ifdef USE_LAPACK
#include <clapack.h>
#endif

PyMODINIT_FUNC init_algorithms(void);
static PyObject *algorithms_kmeans(PyObject *self, PyObject *args);

#ifdef USE_LAPACK
static PyObject *algorithms_solve_system(PyObject *self, PyObject *args);
static PyObject *algorithms_lu_solve(PyObject *self, PyObject *args);
static PyObject *algorithms_em(PyObject *self, PyObject *args);
static PyObject *algorithms_log_multi_gauss(PyObject *self, PyObject *args);
#endif

static PyMethodDef module_methods[] = {
    {"kmeans", algorithms_kmeans, METH_VARARGS, "Faster K-means."},
#ifdef USE_LAPACK
    {"solve_system", algorithms_solve_system, METH_VARARGS, "Solve a system of equations."},
    {"lu_solve", algorithms_lu_solve, METH_VARARGS, "Compute the LU factorization."},
    {"em", algorithms_em, METH_VARARGS, "Faster EM."},
    {"log_multi_gauss", algorithms_log_multi_gauss, METH_VARARGS, "Log-multi-Gaussian."},
#endif
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC init_algorithms(void)
{
    PyObject *m = Py_InitModule("_algorithms", module_methods);
    if (m == NULL)
        return;
    import_array(); /* Load NumPy */
}

static PyObject *algorithms_kmeans(PyObject *self, PyObject *args)
{
    /* SHAPES:
        data  -> (P, D)
        means -> (K, D)
        rs    -> (P,)
     */

    /* parse the input tuple */
    PyObject *data_obj = NULL, *means_obj = NULL, *rs_obj = NULL;
    double tol;
    int maxiter;
    if (!PyArg_ParseTuple(args, "OOOdi", &data_obj, &means_obj, &rs_obj, &tol, &maxiter))
        return NULL;

    /* get numpy arrays */
    PyObject *data_array  = PyArray_FROM_OTF(data_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *means_array = PyArray_FROM_OTF(means_obj, NPY_DOUBLE, NPY_INOUT_ARRAY);
    PyObject *rs_array    = PyArray_FROM_OTF(rs_obj, NPY_INTP, NPY_INOUT_ARRAY);
    if (data_array == NULL || means_array == NULL || rs_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "input objects can't be converted to arrays.");
        Py_XDECREF(data_array);
        Py_XDECREF(means_array);
        Py_XDECREF(rs_array);
        return NULL;
    }

    double *data  = (double*)PyArray_DATA(data_array);
    double *means = (double*)PyArray_DATA(means_array);
    long   *rs    = (long*)PyArray_DATA(rs_array);

    int p, d, k;
    int P = (int)PyArray_DIM(data_array, 0);
    int D = (int)PyArray_DIM(data_array, 1);
    int K = (int)PyArray_DIM(means_array, 0);

    double *dists = (double*)malloc(K*sizeof(double));
    long   *N_rs  = (long*)malloc(K*sizeof(long));

    double L = 1.0;
    int iter;
    for (iter = 0; iter < maxiter; iter++) {
        double L_new = 0.0, dL;
        for (p = 0; p < P; p++) {
            double min_dist = -1.0;
            for (k = 0; k < K; k++) {
                dists[k] = 0.0;
                for (d = 0; d < D; d++) {
                    double diff = means[k*D+d] - data[p*D+d];
                    dists[k] += diff*diff;
                }
                if (min_dist < 0 || dists[k] < min_dist) {
                    min_dist = dists[k];
                    rs[p] = k;
                }
            }
            L_new += dists[rs[p]];
        }

        /* check for convergence */
        dL = fabs(L_new - L)/L;
        if (iter > 5 && dL < tol)
            break;
        else
            L = L_new;

        /* update means */
        for (k = 0; k < K; k++)
            N_rs[k] = 0;
        for (p = 0; p < P; p++) {
            N_rs[rs[p]] += 1;

            for (d = 0; d < D; d++) {
                means[rs[p]*D + d] += data[p*D + d];
            }
        }

        for (k = 0; k < K; k++) {
            for (d = 0; d < D; d++) {
                means[k*D + d] /= (double)N_rs[k];
            }
        }
    }

    if (iter < maxiter)
        printf("K-means converged after %d iterations\n", iter);
    else
        printf("K-means didn't converge after %d iterations\n", iter);

    /* clean up */
    Py_DECREF(data_array);
    Py_DECREF(means_array);
    Py_DECREF(rs_array);
    free(dists);
    free(N_rs);

    /* return None */
    Py_INCREF(Py_None);
    return Py_None;
}

#ifdef USE_LAPACK

int solve_system(double *a, double *b, int dim_a, int dim_b)
{
    int *piv = (int*)malloc(dim_a * sizeof(int));
    int info;

    dgesv_( &dim_a, &dim_b, a, &dim_a, piv, b, &dim_a, &info );

    free(piv);

    return info;
}

static PyObject *algorithms_solve_system(PyObject *self, PyObject *args)
{
    /* parse the input tuple */
    PyObject *a_obj = NULL, *b_obj = NULL;
    if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj))
        return NULL;

    /* get numpy arrays */
    PyObject *a_array = PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_INOUT_ARRAY);
    PyObject *b_array = PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_INOUT_ARRAY);
    if (a_array == NULL || b_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "input objects can't be converted to arrays.");
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        return NULL;
    }

    double *a = (double*)PyArray_DATA(a_array);
    double *b = (double*)PyArray_DATA(b_array);

    int dim_a = (int)PyArray_DIM(a_array, 0);
    int dim_b = 1;

    if (PyArray_NDIM(b_array) > 1)
        dim_b = PyArray_DIM(b_array, 1);

    int ret = solve_system(a, b, dim_a, dim_b);

    Py_DECREF(a_array);
    Py_DECREF(b_array);

    Py_INCREF(Py_None);
    return Py_None;
}

int lu_factor(double *a, int dim, int *piv)
{
    int info;
    dgetrf_(&dim, &dim, a, &dim, piv, &info);
    return info;
}

int lu_solve(double *a, double *b, int dim_a, int dim_b, int *piv)
{
    int info;
    char t = 'T';
    dgetrs_(&t, &dim_a, &dim_b, a, &dim_a, piv, b, &dim_a, &info);
    return info;
}

double lu_slogdet(double *a, int dim, int *piv, int *s)
{
    int i;
    double logdet = 0.0;
    *s = 1;

    for (i = 0; i < dim; i++) {
        double uii     = a[i*dim + i];
        double abs_uii = fabs(uii);
        if (piv[i] != i+1) /* fortran style indexing */
            *s *= -1;
        if (uii < 0)
            *s *= -1;
        logdet += log(abs_uii);
    }

    return logdet;
}

static PyObject *algorithms_lu_solve(PyObject *self, PyObject *args)
{
    int i, j;

    /* parse the input tuple */
    PyObject *a_obj = NULL, *b_obj = NULL;
    if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj))
        return NULL;

    /* get numpy arrays */
    PyObject *a_array = PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_INOUT_ARRAY);
    PyObject *b_array = PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_INOUT_ARRAY);
    if (a_array == NULL || b_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "input objects can't be converted to arrays.");
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        return NULL;
    }

    /* get pointers to the array data */
    double *a = (double*)PyArray_DATA(a_array);
    double *b = (double*)PyArray_DATA(b_array);

    /* array dimensions */
    int dim_a = (int)PyArray_DIM(a_array, 0);
    int dim_b = 1;
    if (PyArray_NDIM(b_array) > 1)
        dim_b = PyArray_DIM(b_array, 1);

    /* do the LUP factorization */
    int *piv  = (int*)malloc(dim_a * sizeof(int));
    int ret  = lu_factor(a, dim_a, piv);

    /* catch the errors */
    if (ret != 0) {
        if (ret > 0)
            PyErr_SetString(PyExc_RuntimeError, "singular matrix");
        else
            PyErr_SetString(PyExc_RuntimeError, "illegal value");

        free(piv);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    /* calculate the determinant */
    int s;
    double logdet = lu_slogdet(a, dim_a, piv, &s);
    double det = ((double)s)*exp(logdet);

    /* solve the system */
    ret = lu_solve(a, b, dim_a, dim_b, piv);

    /* catch the errors */
    if (ret != 0) {
        PyErr_SetString(PyExc_RuntimeError, "illegal value");

        free(piv);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    /* clean up */
    free(piv);
    Py_DECREF(a_array);
    Py_DECREF(b_array);

    Py_INCREF(Py_None);
    return Py_None;
}

double log_multi_gauss(double *x, double *mu, double *lu, int *piv, double logdet, int dim, int *info)
{
    int i;
    double result = 0.0;
    double *X = (double*)malloc(dim*sizeof(double));
    double *Y = (double*)malloc(dim*sizeof(double));
    for (i = 0; i < dim; i++)
        X[i] = Y[i] = x[i]-mu[i];
    *info = lu_solve(lu, Y, dim, 1, piv);

    if (*info != 0)
        return 0;

    for (i = 0; i < dim; i++)
        result += X[i]*Y[i];
    result *= -0.5;
    result += - 0.5 * (logdet + dim*log(2*M_PI));
    free(X); free(Y);
    return result;
}

static PyObject *algorithms_log_multi_gauss(PyObject *self, PyObject *args)
{
    int i, j;

    /* parse the input tuple */
    PyObject *a_obj, *b_obj, *c_obj;
    if (!PyArg_ParseTuple(args, "OOO", &c_obj, &b_obj, &a_obj))
        return NULL;

    /* get numpy arrays */
    PyObject *a_array = PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_INOUT_ARRAY);
    PyObject *b_array = PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_INOUT_ARRAY);
    PyObject *c_array = PyArray_FROM_OTF(c_obj, NPY_DOUBLE, NPY_INOUT_ARRAY);
    if (a_array == NULL || b_array == NULL || c_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "input objects can't be converted to arrays.");
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        return NULL;
    }

    /* get pointers to the array data */
    double *a = (double*)PyArray_DATA(a_array);
    double *b = (double*)PyArray_DATA(b_array);
    double *c = (double*)PyArray_DATA(c_array);

    /* array dimensions */
    int dim_a = (int)PyArray_DIM(a_array, 0);

    /* do the LUP factorization */
    int *piv  = (int*)malloc(dim_a * sizeof(int));
    int ret  = lu_factor(a, dim_a, piv);

    /* catch the errors */
    if (ret != 0) {
        if (ret > 0)
            PyErr_SetString(PyExc_RuntimeError, "singular matrix");
        else
            PyErr_SetString(PyExc_RuntimeError, "illegal value");

        free(piv);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    /* calculate the determinant */
    int s, info;
    double logdet = lu_slogdet(a, dim_a, piv, &s);
    double det = ((double)s)*exp(logdet);

    double multi_gauss = log_multi_gauss(c, b, a, piv, logdet, dim_a, &info);

    PyObject *result = Py_BuildValue("d", multi_gauss);

    /* clean up */
    free(piv);
    Py_DECREF(a_array);
    Py_DECREF(b_array);

    Py_INCREF(result);
    return result;
}

double log_sum_exp(double a, double b)
{
    if (a > b)
        return a + log(1+exp(b-a));
    return b + log(1+exp(a-b));
}

static PyObject *algorithms_em(PyObject *self, PyObject *args)
{
    /* SHAPES:
        data  -> (P, D)
        means -> (K, D)
        rs    -> (P,)
     */

    /* parse the input tuple */
    PyObject *data_obj, *means_obj, *alphas_obj, *cov_obj;
    double tol;
    int maxiter;
    if (!PyArg_ParseTuple(args, "OOOOdi", &data_obj, &means_obj, &cov_obj,
                &alphas_obj, &tol, &maxiter))
        return NULL;

    /* get numpy arrays */
    PyObject *data_array   = PyArray_FROM_OTF(data_obj,  NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *means_array  = PyArray_FROM_OTF(means_obj, NPY_DOUBLE, NPY_INOUT_ARRAY);
    PyObject *alphas_array = PyArray_FROM_OTF(alphas_obj,NPY_DOUBLE, NPY_INOUT_ARRAY);
    PyObject *cov_array    = PyArray_FROM_OTF(cov_obj,   NPY_DOUBLE, NPY_INOUT_ARRAY);
    if (data_array == NULL || means_array == NULL || alphas_array == NULL
            || cov_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "input objects can't be converted to arrays.");
        Py_XDECREF(data_array);
        Py_XDECREF(means_array);
        Py_XDECREF(alphas_array);
        Py_XDECREF(cov_array);
        return NULL;
    }

    double *data   = (double*)PyArray_DATA(data_array);
    double *means  = (double*)PyArray_DATA(means_array);
    double *alphas = (double*)PyArray_DATA(alphas_array);
    double *cov    = (double*)PyArray_DATA(cov_array);

    int i, p, d, k;
    int P = (int)PyArray_DIM(data_array, 0);
    int D = (int)PyArray_DIM(data_array, 1);
    int K = (int)PyArray_DIM(means_array, 0);

    double *loggammas = (double*)malloc(K*P*sizeof(double));
    double *logNk     = (double*)malloc(K*sizeof(double));

    double *lu     = (double*)malloc(K*D*D*sizeof(double));
    double *logdet = (double*)malloc(K*sizeof(double));
    int    *piv    = (int*)   malloc(K*D*sizeof(int));

    for (k = 0; k < K; k++)
        alphas[k] = log(alphas[k]);

    double L = 1.0;
    int iter;
    for (iter = 0; iter < maxiter; iter++) {
        double L_new = 0.0, dL;

        /* pre-factorization */
        for (i = 0; i < K*D*D; i++)
            lu[i] = cov[i];
        for (k = 0; k < K; k++) {
            int ret = lu_factor(&lu[k*D*D], D, &piv[k*D]);
            int s;
            logdet[k] = lu_slogdet(&lu[k*D*D], D, &piv[k*D], &s);

            if (ret != 0 || s <= 0) {
                if (ret != 0)
                    PyErr_SetString(PyExc_RuntimeError, "couldn't factorize cov");
                else
                    PyErr_SetString(PyExc_RuntimeError, "cov must be positive definite");

                free(loggammas);
                free(logNk);
                free(lu);
                free(logdet);
                free(piv);
                Py_DECREF(data_array);
                Py_DECREF(means_array);
                Py_DECREF(alphas_array);
                Py_DECREF(cov_array);
                return NULL;
            }
        }

        /* Expectation step */
        for (p = 0; p < P; p++) {
            double logmu = 0.0, logprob = 0.0;
            for (k = 0; k < K; k++) {
                int info;
                double logNpk = log_multi_gauss(&data[p*D], &means[k*D],
                                            &lu[k*D*D], &piv[k*D], logdet[k],
                                            D, &info);

                if (info != 0) {
                    PyErr_SetString(PyExc_RuntimeError, "couldn't generate multi-gauss");
                    free(loggammas);
                    free(logNk);
                    free(lu);
                    free(logdet);
                    free(piv);
                    Py_DECREF(data_array);
                    Py_DECREF(means_array);
                    Py_DECREF(alphas_array);
                    Py_DECREF(cov_array);
                    return NULL;
                }

                loggammas[p*K+k] = alphas[k] + logNpk;
                if (k == 0)
                    logmu = loggammas[p*K+k];
                else
                    logmu = log_sum_exp(logmu, loggammas[p*K+k]);
            }
            for (k = 0; k < K; k++) {
                loggammas[p*K+k] -= logmu;
                if (p == 0)
                    logNk[k] = loggammas[p*K+k];
                else
                    logNk[k] = log_sum_exp(logNk[k], loggammas[p*K+k]);
            }
            L_new += logmu;
        }

        if (iter == 0)
            printf("Initial log(L) = %f\n", L_new);

        /* check for convergence */
        dL = fabs((L_new - L)/L);
        L = L_new;

        if (iter > 5 && dL < tol)
            break;

        /* Maximization step */
        for (k = 0; k < K; k++) {
            alphas[k] = logNk[k] - log(P);
            for (d = 0; d < D; d++)
                means[k*D+d] = 0.0;
        }

        for (p = 0; p < P; p ++) {
            for (k = 0; k < K; k++) {
                double factor = exp(loggammas[p*K+k] - logNk[k]);
                for (d = 0; d < D; d++)
                    means[k*D+d] += factor * data[p*D+d];
            }
        }

        for (i = 0; i < K*D*D; i++)
            cov[i] = 0.0;

        for (p = 0; p < P; p ++) {
            for (k = 0; k < K; k++) {
                double factor = exp(loggammas[p*K+k] - logNk[k]);
                for (d = 0; d < D; d++) {
                    for (i = d; i < D; i++) {
                        cov[k*D*D + d*D + i] += factor
                            * (data[p*D+d] - means[k*D+d])
                            * (data[p*D+i] - means[k*D+i]);
                        if (i == d)
                            cov[k*D*D + d*D + i] += 1.0e-10;
                    }
                }
            }
        }
        for (k = 0; k < K; k++)
            for (d = 0; d < D; d++)
                for (i = d+1; i < D; i++)
                    cov[k*D*D + i*D + d] = cov[k*D*D + d*D + i];
    }

    if (iter < maxiter)
        printf("EM converged after %d iterations\nFinal log(L) = %f\n", iter, L);
    else
        printf("EM didn't converge after %d iterations\n", iter);

    for (k = 0; k < K; k++)
        alphas[k] = exp(alphas[k]);

    /* clean up */
    free(loggammas);
    free(logNk);
    free(lu);
    free(logdet);
    free(piv);
    Py_DECREF(data_array);
    Py_DECREF(means_array);
    Py_DECREF(alphas_array);
    Py_DECREF(cov_array);

    /* return None */
    Py_INCREF(Py_None);
    return Py_None;
}

#endif /* USE_LAPACK */

