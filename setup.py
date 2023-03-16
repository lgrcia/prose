from pathlib import Path
from setuptools import find_packages, setup

HERE = Path(__file__).parent
README = (HERE / "README.md").read_text(encoding="utf-8")

setup(
    name="prose",
    version="3.0.0",
    author="Lionel J. Garcia",
    description="Reduction and analysis of FITS telescope observations",
    packages=find_packages(exclude=["test"]),
    include_package_data=True,
    license="MIT",
    url="https://github.com/lgrcia/prose",
    # entry_points="""
    #     [console_scripts]
    #     prose=main:cli
    # """,
    long_description=README,
    long_description_content_type="text/markdown",
    install_requires=[
        "ipython",
        "numpy",
        "ipython",
        "scipy",
        "astropy",
        "matplotlib",
        "scikit-image",
        "pandas>=1.1",
        "tqdm",
        "photutils",
        "astroquery",
        "pyyaml",
        "tabulate",
        "requests",
        "imageio[ffmpeg]",
        "sep",
        "netcdf4",
        "celerite2",
        "jinja2",
        "twirl",
        "multiprocess",
    ],
    extras_require={
        "docs": [
            "sphinx",
            "docutils",
            "jupyterlab",
            "myst-parser",
            "twine",
            "sphinx-book-theme",
            "black",
            "myst_nb",
            "sphinx-copybutton",
            "jupyter",
        ]
    },
    zip_safe=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
