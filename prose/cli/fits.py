import argparse
import shutil
from pathlib import Path

import pandas as pd

from prose import FITSImage
from prose.io import FitsManager


def pick_observation_id(observations):
    if len(observations) == 0:
        print("No observations found")
        return
    elif len(observations) == 1:
        observation_id = observations.index[0]
    else:
        print(f"{len(observations)} observations found:")
        print(observations, "\n")
        while observation_id is None:
            print(f"Which observation id do you want to reduce?")
            observation_id = input()
            if not int(observation_id) in observations.index.values:
                print("Invalid observation id")
                observation_id = None

    return int(observation_id)


def fits(args):
    fm = FitsManager(folders=args.folder, file=args.file, depth=args.depth, leave=False)
    observations = fm.observations()
    print(observations)


def add_fits_parser(subparsers):
    fits_parser = subparsers.add_parser(
        name="fits", description="parse and store data from FITS in folder(s)"
    )
    fits_parser.add_argument("folder", type=str, help="folder to explore", default=".")
    fits_parser.add_argument(
        "-d", "--depth", type=int, help="depth of the search", default=10
    )
    fits_parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="SQLite database file to save parsing results",
        default=None,
    )
    fits_parser.set_defaults(func=fits)


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


def add_db_parser(subparsers):
    db_parser = subparsers.add_parser(name="db", description="explore a FITS database")
    db_parser.add_argument(
        "file", type=str, help="SQLite database file to explore", default=None
    )
    db_parser.add_argument(
        "-m", "--max-rows", type=int, help="max number of rows to display", default=50
    )
    db_parser.add_argument(
        "-t", "--telescope", type=str, help="telescope name to filter for", default=None
    )
    db_parser.add_argument(
        "-f", "--filter", type=str, help="filter name to filter for", default=None
    )
    db_parser.add_argument(
        "-d", "--date", type=str, help="observation date to filter for", default=None
    )
    db_parser.add_argument(
        "-o", "--target", type=str, help="target name to filter for", default=None
    )
    db_parser.set_defaults(func=db)


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


def add_info_parser(subparsers):
    info_parser = subparsers.add_parser(
        name="info", description="print FITS image information"
    )
    info_parser.add_argument("filename", type=str, help="FITS image filename")
    info_parser.set_defaults(func=info)


# experimental
# ------------
def organize(args):
    fm = FitsManager(
        folders=args.folder,
        depth=args.depth,
        leave=False,
    )
    new_folders = {}

    main_folder = Path(args.folder) if args.output is None else Path(args.output)
    obs = fm.observations(type="light")

    if args.separate:
        for observation_id, observation in obs.iterrows():
            date_str = observation.date.replace("-", "_")
            new_folders[date_str] = fm.observation_files(observation_id, show=False)

    print(f"\nprose will create the following folders (in {main_folder}):")

    for dates in new_folders.keys():
        # sum all len file files in folder dict
        n_files = sum([len(files) for files in new_folders.values()])
        print("-", dates, f"({n_files} files)")

    proceed = input("\ncontinue? [y]/n: ")
    if proceed == "n":
        return
    else:
        for date, files in new_folders.items():
            date_folder = main_folder / date
            date_folder.mkdir(parents=True, exist_ok=True)
            for filetype, files in files.items():
                im_type = date_folder / filetype
                im_type.mkdir(parents=True, exist_ok=True)
                for file in files:
                    if im_type == "images":
                        shutil.move(file, im_type / Path(file).name)
                    else:
                        try:
                            shutil.copy2(file, im_type / Path(file).name)
                        except shutil.SameFileError:
                            pass


def add_organize_parser(subparsers):
    organize_parser = subparsers.add_parser(
        name="organize", description="organize FITS files by date"
    )
    organize_parser.add_argument(
        "folder", type=str, help="folder to explore", default="."
    )
    organize_parser.add_argument(
        "-d", "--depth", type=int, help="depth of the search", default=10
    )
    organize_parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="folder to store organized files",
        default=None,
    )
    organize_parser.add_argument(
        "-s",
        "--separate",
        action="store_true",
        help="separate each calibration file into its respective observation folder",
    )
    organize_parser.set_defaults(func=organize)
