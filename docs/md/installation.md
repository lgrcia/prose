
# Installation

prose is written for python 3 and can be installed in different ways (all relying on pip)

## conda

To install it in a fresh conda environment
```shell
conda env create -f {prose_repo}/environment.yml -n prose
```

## pip

```shell
pip install prose
```

## from source

`git clone` prose repository with 

```shell
git clone https://github.com/lgrcia/prose.git
```

then locally install the package with

```shell 
pip install -e {prose_repo}
```