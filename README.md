# prose

prose is a python package to reduce and analyse data from telescope observations. Its primary goal is the production of differential light curves from raw uncalibrated FITS images.

<p align="center">
  <img width="400" src="docs/source/prose.png">
</p>

```python

from prose import FitsManager

fm = FitsManager("folder")
fm.describe()
```
```

    ╒════════════╤══════════════╤════════╤═════════════╤══════════════╤══════════╤════════════╕
    │ date       │ telescope    │ type   │ target      │ dimensions   │ filter   │   quantity │
    ╞════════════╪══════════════╪════════╪═════════════╪══════════════╪══════════╪════════════╡
    │ 2019-09-22 │ SPECULOOS-IO │ bias   │             │ 2048x2088    │          │          9 │
    ├────────────┼──────────────┼────────┼─────────────┼──────────────┼──────────┼────────────┤
    │ 2019-09-22 │ SPECULOOS-IO │ dark   │             │ 2048x2088    │          │         27 │
    ├────────────┼──────────────┼────────┼─────────────┼──────────────┼──────────┼────────────┤
    │ 2019-09-22 │ SPECULOOS-IO │ flat   │             │ 2048x2088    │ I+z      │         14 │
    ├────────────┼──────────────┼────────┼─────────────┼──────────────┼──────────┼────────────┤
    │ 2019-09-22 │ SPECULOOS-IO │ light  │ Sp0111-4908 │ 2048x2088    │ I+z      │        263 │
    ╘════════════╧══════════════╧════════╧═════════════╧══════════════╧══════════╧════════════╛
```

```python

from prose import pipeline

reduction = pipeline.Reduction("folder")
destination = reduction.run()

photometry = pipeline.Photometry(destination)
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