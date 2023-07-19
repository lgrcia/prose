import argparse
from pathlib import Path

import pandas as pd

from prose import FITSImage
from prose.io import FitsManager


def fits(args):
    fm = FitsManager(folders=args.folder, file=args.file)
    observations = fm.observations()
    print(observations)


def db(args):
    # observations
    # ------------
    fm = FitsManager(file=args.file)
    observations = fm.observations(
        telescope=args.telescope,
        filter=args.filter,
        date=args.date,
        target=args.target,
    )
    print(observations)


def info(args):
    image = FITSImage(args.filename)
    print(f"filename: {Path(args.filename).stem}")
    print(f"telescope: {image.telescope.name}")
    print(f"date: {image.date}")
    print(f"target: {image.metadata['object']}")
    print(f"filter: {image.filter}")
    print(f"exposure: {image.exposure}")
    print(f"dimensions: {image.shape}")
    print(f"JD: {image.jd}")
    print(f"RA: {image.ra}")
    print(f"DEC: {image.dec}")
    print(f"pixel scale: {image.pixel_scale}")
