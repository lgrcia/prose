import argparse
from pathlib import Path

from prose import FitsManager, Sequence, blocks


def calibrate(args):
    fm = FitsManager(args.folder, depth=args.depth)
    observations = fm.observations(type="light")
    observation_id = None

    # observation selection
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
    print("\n")
    folder = Path(args.folder)
    calibrated_folder = Path(str(folder.absolute()) + "_calibrated")
    calibrated_folder.mkdir(exist_ok=True)
    observation_id = int(observation_id)

    files = fm.observation_files(observation_id, show=False)
    darks = files["darks"]
    flats = files["flats"]
    bias = files["bias"]
    lights = files["images"]

    # calibration
    calibration = Sequence(
        [
            blocks.Calibration(darks=darks, flats=flats, bias=bias),
            blocks.Trim(),
            blocks.WriteTo(calibrated_folder, label="calibrated"),
        ],
        name="Calibration",
    )

    calibration.run(lights)
    print("Calibrated images saved in", calibrated_folder)


def add_calibrate_parser(subparsers):
    calibrate_parser = subparsers.add_parser(
        name="calibrate", description="calibrate FITS files"
    )
    calibrate_parser.add_argument(
        "folder",
        type=str,
        help="folder to parse containing science and calibration files",
        default=None,
    )
    calibrate_parser.add_argument(
        "-d", "--depth", type=int, help="subfolder parsing depth", default=10
    )
    calibrate_parser.set_defaults(func=calibrate)
