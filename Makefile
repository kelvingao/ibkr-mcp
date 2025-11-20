PYTHON ?= python

.PHONY: build clean twine

build:
	$(PYTHON) -m build

clean:
	rm -rf build dist *.egg-info

twine: clean build
	twine upload dist/*

