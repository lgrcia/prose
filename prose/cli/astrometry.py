import argparse
from pathlib import Path

from prose import FITSImage, FitsManager, Sequence, blocks


def solve(args):
    image = FITSImage(args.file)

    solve_sequence = Sequence([blocks.PointSourceDetection(n=30), blocks.PlateSolve()])

    solve_sequence.run(image)
    image.writeto(args.file)


def add_solve_parser(subparsers):
    solve_parser = subparsers.add_parser(name="solve", description="solve FITS image")
    solve_parser.add_argument("file", type=str, help="file to solve", default=None)
    solve_parser.set_defaults(func=solve)
