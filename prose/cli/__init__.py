import argparse

import click
import yaml

from prose.cli.astrometry import add_solve_parser
from prose.cli.calibration import add_calibrate_parser
from prose.cli.fits import (
    add_db_parser,
    add_fits_parser,
    add_info_parser,
    add_organize_parser,
)
from prose.cli.stack import stack
from prose.cli.visualisation import show, video


@click.group()
def main():
    """
    \b
    ░▄▀▀▄░█▀▀▄░▄▀▀▄░█▀▀░█▀▀  *  .
    ░█▄▄█░█▄▄▀░█░░█░▀▀▄░█▀▀   .*
    ░█░░░░▀░▀▀░░▀▀░░▀▀▀░▀▀▀ +
    """
    pass


main.add_command(show)
main.add_command(video)
main.add_command(stack)
