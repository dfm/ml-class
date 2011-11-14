all: dataset

clean:
	rm -f dataset/_tile_herlper.so

dataset: dataset/_tile_helper.c
	python setup.py build_ext --inplace
