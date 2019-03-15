all:
    # install ochumanApi locally
	python setup.py build_ext --inplace
	rm -rf build

install:
	# install ochumanApi to the Python site-packages
	python setup.py build_ext install
	rm -rf build
