#include <Python.h>
#include <numpy/arrayobject.h>
#include <clapack.h>

PyMODINIT_FUNC init_algorithms(void);
static PyObject *algorithms_kmeans(PyObject *self, PyObject *args);
static PyObject *algorithms_solve_system(PyObject *self, PyObject *args);
static PyObject *algorithms_lu_solve(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
    {"kmeans", algorithms_kmeans, METH_VARARGS, "Faster K-means."},
    {"solve_system", algorithms_solve_system, METH_VARARGS, "Solve a system of equations."},
    {"lu_solve", algorithms_lu_solve, METH_VARARGS, "Compute the LU factorization."},
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
            int min_k = 0;
            for (k = 0; k < K; k++) {
                dists[k] = 0.0;
                for (d = 0; d < D; d++) {
                    double diff = means[k*D+d] - data[p*D+d];
                    dists[k] += diff*diff;
                }
                // printf("%f\n",dists[ind]);
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

double lu_det(double *a, int dim, int *piv)
{
    int i;
    double det = 1.0;

    for (i = 0; i < dim; i++) {
        if (piv[i] != i+1) /* fortran style indexing */
            det *= -1.0;
        det *= a[i*dim + i];
    }

    return det;
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
    double det = lu_det(a, dim_a, piv);

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


