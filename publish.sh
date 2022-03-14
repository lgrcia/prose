#!/bin/sh

rm -r dist
python setup.py sdist bdist_wheel
tar tzf dist/*.tar.gz
twine upload dist/*

