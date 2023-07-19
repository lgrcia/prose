import argparse

import yaml

from prose.cli.calibration import calibrate
from prose.cli.stack import stack
from prose.cli.visualisation import show


def make_parser():
    main_parser = argparse.ArgumentParser(prog="prose", description="prose")
    subparsers = main_parser.add_subparsers(required=True)

    # calibrate
    # ---------
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

    # show
    # ----
    show_parser = subparsers.add_parser(name="show", description="show FITS image")
    show_parser.add_argument("file", type=str, help="file to show", default=None)
    show_parser.add_argument(
        "-c",
        "--contrast",
        type=float,
        help="contrast of the image (zscale is applied)",
        default=0.1,
    )
    show_parser.set_defaults(func=show)

    # stack
    # -----
    stack_parser = subparsers.add_parser(name="stack", description="stack FITS files")
    stack_parser.add_argument("folder", type=str, help="folder to parse", default=None)
    stack_parser.add_argument(
        "-d", "--depth", type=int, help="subfolder parsing depth", default=10
    )
    stack_parser.add_argument(
        "-n", "--n", type=int, help="number of stars used for alignment", default=30
    )
    stack_parser.add_argument(
        "--method",
        choices=["mean", "selective"],
        help="alignment method. 'mean' applies a mean to all images, 'selective' \
            applies a median to the -n smallest-FWHM images",
        default="mean",
    )
    stack_parser.set_defaults(func=stack)

    # fitsmanager
    # -----------

    # epsf
    # ----

    return main_parser


def to_yaml(parser, output_file):
    def parse_arguments(action):
        argument_info = {
            "name": action.dest,
            "short": action.option_strings[0] if action.option_strings else None,
            "long": action.option_strings[1]
            if len(action.option_strings) > 1
            else None,
            "type": action.type.__name__ if action.type else None,
            "default": action.default,
            "required": action.required,
            "help": action.help,
            "choices": action.choices,
            "nargs": action.nargs,
        }
        return {k: v for k, v in argument_info.items() if v is not None}

    def parse_subparsers(action_group):
        commands = {}
        for action in action_group._group_actions:
            if isinstance(action, argparse._SubParsersAction):
                for subparser_name, subparser_obj in action.choices.items():
                    command_info = {
                        "help": subparser_obj.description,
                        "arguments": [
                            parse_arguments(action)
                            for action in subparser_obj._actions
                            if not isinstance(action, argparse._HelpAction)
                        ],
                    }
                    commands[subparser_name] = command_info
            else:
                # Handle nested subparsers recursively
                commands.update(parse_subparsers(action))
        return commands

    cli_info = {
        "commands": parse_subparsers(parser._subparsers),
    }

    with open(output_file, "w") as yaml_file:
        yaml.dump(cli_info, yaml_file, default_flow_style=False)


def main():
    main_parser = make_parser()
    args = main_parser.parse_args()
    args.func(args)
