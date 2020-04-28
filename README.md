# prose

prose is a python package to reduce and analyse data from telescope observations. Its primary goal is the production of differential light curves from raw uncalibrated FITS images.

<p align="center">
  <img width="400" src="docs/source/_static/css/prose.png">
</p>

## Installation

prose runs more safely in its own [virtual environment](https://docs.python.org/3/tutorial/venv.html) and is tested on Python 3.6.

### OSX

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