# prose

A framework for FITS processing pipelines in python. Built for Astronomy, *prose* features pipelines to perform common tasks (such as automated calibration, reduction and photometry) and make building custom ones easy. Documentation at [prose.readthedocs.io](https://prose.readthedocs.io)

<p align="center">
  <img width="400" src="docs/source/prose.png">
</p>


```python

from prose import Reduction, AperturePhotometry

reduction = Reduction(fits_folder)
reduction.run()

photometry = AperturePhotometry(reduction.destination)
photometry.run()

```

## Installation

prose runs more safely in its own [virtual environment](https://docs.python.org/3/tutorial/venv.html) and is tested on Python 3.6.

### example on OSX

create your [virtualenv](https://docs.python.org/3/tutorial/venv.html) and activate it

```shell
python3.6 -m venv prose_env
source prose_env/bin/activate.bin
```

Then to locally install prose

```shell
git clone https://github.com/LionelGarcia/prose.git

cd prose_env
python3.6 -m pip install -e ../prose
```

Applicable to Linux-based and Windows OS