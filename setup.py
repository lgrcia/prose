from pathlib import Path
from setuptools import setup

HERE = Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name="prose",
    version="0.9.33",
    author="Lionel J. Garcia",
    description="Reduction and analysis of FITS telescope observations",
    packages=["prose"],
    package_dir={"": "prose"},
    license="MIT",
    url="https://github.com/lgrcia/prose",
    # entry_points="""
    #     [console_scripts]
    #     prose=main:cli
    # """,
    long_description=README,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy",
        "scipy",
        "astropy==4.0",
        "matplotlib",
        "colorama",
        "scikit-image",
        "pandas",
        "tqdm",
        "astroalign",
        "photutils",
        "astroquery",
        "pyyaml",
        "sphinx",
        "nbsphinx",
        "docutils",
        "tabulate",
        "requests",
        "sphinx_rtd_theme",
        "imageio",
        "sep",
        "xarray",
        "numba",
        "netcdf4",
        "nbsphinx",
        "celerite2",
        "jinja2",
        "tensorflow"
    ],
    zip_safe=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
