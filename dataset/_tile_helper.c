#include <Python.h>
#include <numpy/arrayobject.h>

PyMODINIT_FUNC init_tile_helper(void);
static PyObject *tile_helper_to_tiles(PyObject *self, PyObject *args);
static PyObject *tile_helper_from_tiles(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
    {"to_tiles", tile_helper_to_tiles, METH_VARARGS, "Convert to tiles."},
    {"from_tiles", tile_helper_from_tiles, METH_VARARGS,
        "Generate an image from a set of prototypes and a list of memberships."},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC init_tile_helper(void)
{
    PyObject *m = Py_InitModule("_tile_helper", module_methods);
    if (m == NULL)
        return;
    import_array(); /* Load NumPy */
}

static PyObject *tile_helper_to_tiles(PyObject *self, PyObject *args)
{
    /* parse the input tuple */
    PyObject *img_in = NULL, *img_out = NULL;
    int tile_x, tile_y;
    if (!PyArg_ParseTuple(args, "OO(ii)", &img_in, &img_out, &tile_x, &tile_y))
        return NULL;
    PyObject *obj_in  = PyArray_FROM_OTF(img_in, NPY_INTP, NPY_IN_ARRAY);
    PyObject *obj_out = PyArray_FROM_OTF(img_out, NPY_INTP, NPY_OUT_ARRAY);
    if (obj_in == NULL || obj_out == NULL) {
        PyErr_SetString(PyExc_TypeError, "input objects can't be converted to arrays.");
        Py_XDECREF(obj_in);
        Py_XDECREF(obj_out);
        return NULL;
    }

    /* array specs */
    int dim_in_x = (int)PyArray_DIM(obj_in, 0), dim_in_y = (int)PyArray_DIM(obj_in, 1);
    int tile_size = tile_x*tile_y;

    /* array data pointers */
    long *data_in  = (long*)PyArray_DATA(obj_in);
    long *data_out = (long*)PyArray_DATA(obj_out);

    /* tiles dim */
    int n_tiles_y = dim_in_y/tile_y;

    /* split into tiles here */
    int xi, yi;
    for (xi = 0; xi < dim_in_x; xi++) {
        for (yi = 0; yi < dim_in_y; yi++) {
            int t_x = xi/tile_x,     t_y = yi/tile_y; /* integer division */
            int n_x = xi-t_x*tile_x, n_y = yi-t_y*tile_y;
            int j = (t_x*n_tiles_y + t_y)*tile_size + n_x*tile_y + n_y;
            data_out[j] = data_in[xi*dim_in_y + yi];
        }
    } /* </MAGIC> */

    /* clean up */
    Py_DECREF(obj_in);
    Py_DECREF(obj_out);

    /* return None */
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *tile_helper_from_tiles(PyObject *self, PyObject *args)
{
    /* parse the input tuple */
    PyObject *means_obj = NULL, *inds_obj = NULL, *img_obj = NULL;
    int tile_x, tile_y;
    if (!PyArg_ParseTuple(args, "OOO(ii)", &means_obj, &inds_obj, &img_obj, &tile_x, &tile_y))
        return NULL;
    PyObject *means_array = PyArray_FROM_OTF(means_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *inds_array  = PyArray_FROM_OTF(inds_obj, NPY_INTP, NPY_IN_ARRAY);
    PyObject *img_array   = PyArray_FROM_OTF(img_obj, NPY_DOUBLE, NPY_OUT_ARRAY);
    if (means_array == NULL || inds_array == NULL || img_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "input objects can't be converted to arrays.");
        Py_XDECREF(means_array);
        Py_XDECREF(inds_array);
        Py_XDECREF(img_array);
        return NULL;
    }

    /* array specs */
    int tile_size = tile_x*tile_y;
    int dim_x = (int)PyArray_DIM(img_array, 0),
        dim_y = (int)PyArray_DIM(img_array, 1);
    int n_tiles_y = dim_y/tile_y;

    /* array data pointers */
    double *means_data = (double*)PyArray_DATA(means_array);
    long *inds_data  = (long*)PyArray_DATA(inds_array);
    double *img_data   = (double*)PyArray_DATA(img_array);

    int xi, yi;
    for (xi = 0; xi < dim_x; xi++) {
        for (yi = 0; yi < dim_y; yi++) {
            int t_x = xi/tile_x,     t_y = yi/tile_y;
            int n_x = xi-t_x*tile_x, n_y = yi-t_y*tile_y;
            int j = (inds_data[t_x*n_tiles_y + t_y])*tile_size + n_x*tile_y + n_y;
            img_data[xi*dim_y + yi] = means_data[j];
        }
    }

    /* clean up */
    Py_DECREF(means_array);
    Py_DECREF(inds_array);
    Py_DECREF(img_array);

    /* return None */
    Py_INCREF(Py_None);
    return Py_None;
}

