all: modules

clean:
	rm -f dataset/_tile_herlper.so mixtures/_algorithms.so

modules: dataset/_tile_helper.c mixtures/_algorithms.c
	python setup.py build_ext --inplace
