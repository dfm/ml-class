#include <Python.h>
#include <numpy/arrayobject.h>

PyMODINIT_FUNC init_tile_helper(void);
static PyObject *tile_helper_to_tiles(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
    {"to_tiles", tile_helper_to_tiles, METH_VARARGS, "Convert to tiles."},
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
    int tile_h, tile_w;
    if (!PyArg_ParseTuple(args, "OO(ii)", &img_in, &img_out, &tile_h, &tile_w))
        return NULL;
    PyObject *data_in  = PyArray_FROM_OTF(img_in, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *data_out = PyArray_FROM_OTF(img_out, NPY_DOUBLE, NPY_IN_ARRAY);
    if (data_in == NULL || data_out == NULL) {
        Py_XDECREF(data_in);
        Py_XDECREF(data_out);
        return NULL;
    }

    /* array specs */
    int ndim_in = PyArray_NDIM(data_in);
    int *dims_in = (int*)PyArray_DIMS(data_in);
    int *strides_in = (int*)PyArray_STRIDES(data_in);
    int ndim_out = PyArray_NDIM(data_out);
    int *dims_out = (int*)PyArray_DIMS(data_out);
    int *strides_out = (int*)PyArray_STRIDES(data_out);

    printf("strides_in: ");
    int i;
    for (i = 0; i < ndim_in; i++)
        printf("%d ", (int)PyArray_STRIDE(data_in, i));
    printf("\n");

    /* split into tiles here */
    int xi, yi;
    for (xi = 0; xi < 10; xi += tile_w) {
        for (yi = 0; yi < 10; yi += tile_h) {

        }
    }

    /* clean up */
    Py_DECREF(data_in);
    Py_DECREF(data_out);

    /* return None */
    Py_INCREF(Py_None);
    return Py_None;
}

