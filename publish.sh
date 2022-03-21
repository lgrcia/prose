#!/bin/sh

rm -r dist; rm -r build
python setup.py sdist bdist_wheel
tar tzf dist/*.tar.gz
twine upload dist/*

