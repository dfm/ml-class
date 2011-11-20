all: modules

clean:
	rm -rf build dataset/_tile_helper.so mixtures/_algorithms.so *.pyc dataset/*.pyc mixtures/*.pyc

modules: dataset/_tile_helper.c mixtures/_algorithms.c
	python setup.py build_ext --inplace
