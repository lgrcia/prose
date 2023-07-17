import argparse
from pathlib import Path

import pandas as pd

from prose import FITSImage
from prose.io import FitsManager


def main():
    parser = argparse.ArgumentParser(
        prog="prose FITS manager", description="Explore fits"
    )
    subparsers = parser.add_subparsers(required=True)

    # parse
    # -----
    def _parse(args):
        fm = FitsManager(folders=args.folder, depth=args.depth, file=args.file)
        observations = fm.observations()
        print("Observations:")
        print(observations)

    parse = subparsers.add_parser(name="parse", help="parse a fits folder")
    parse.add_argument("folder", type=str, help="folder to explore", default=".")
    parse.add_argument(
        "-d", "--depth", type=int, help="depth of the search", default=10
    )
    parse.add_argument(
        "-f", "--file", type=str, help="SQLite database file", default=None
    )
    parse.add_argument(
        "-m", "--max-rows", type=int, help="max number of rows to display", default=10
    )
    parse.set_defaults(func=_parse)

    # observations
    # ------------
    def _observations(args):
        fm = FitsManager(file=args.file)
        observations = fm.observations(
            telescope=args.telescope,
            filter=args.filter,
            date=args.date,
            target=args.target,
        )
        print("Observations:")
        print(observations)

    observations = subparsers.add_parser(
        name="observations", help="list observations from an SQLite database"
    )
    observations.add_argument(
        "file", type=str, help="SQLite database file", default=None
    )
    observations.add_argument(
        "-t", "--telescope", type=str, help="telescope name", default=None
    )
    observations.add_argument(
        "-f", "--filter", type=str, help="filter name", default=None
    )
    observations.add_argument("-d", "--date", type=str, help="date", default=None)
    observations.add_argument(
        "-o", "--target", type=str, help="target name", default=None
    )
    observations.add_argument(
        "-m", "--max-rows", type=int, help="max number of rows to display", default=10
    )
    observations.set_defaults(func=_observations)

    # info
    # ----
    def _info(args):
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

    info = subparsers.add_parser(name="info", help="FITS image information")
    info.add_argument("filename", type=str, help="FITS image filename")
    info.set_defaults(func=_info)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
