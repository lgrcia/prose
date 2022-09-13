#!/bin/sh
conda activate prose
cd source/notebooks
jupyter nbconvert --to notebook --execute --inplace archival.ipynb
jupyter nbconvert --to notebook --execute --inplace catalogs.ipynb
jupyter nbconvert --to notebook --execute --inplace fits_manager.ipynb
jupyter nbconvert --to notebook --execute --inplace photometry.ipynb
jupyter nbconvert --to notebook --execute --inplace reports.ipynb
jupyter nbconvert --to notebook --execute --inplace custom_block.ipynb
jupyter nbconvert --to notebook --execute --inplace modeling.ipynb

jupyter nbconvert --to notebook --execute --inplace extra.ipynb

cd ../../
