import argparse

import yaml

from prose.cli.astrometry import add_solve_parser
from prose.cli.calibration import add_calibrate_parser
from prose.cli.fits import (
    add_db_parser,
    add_fits_parser,
    add_info_parser,
    add_organize_parser,
)
from prose.cli.stack import add_stack_parser
from prose.cli.visualisation import add_show_parser, add_video_parser


def make_parser():
    main_parser = argparse.ArgumentParser(prog="prose", description="prose")
    subparsers = main_parser.add_subparsers(required=True)

    add_calibrate_parser(subparsers)
    add_show_parser(subparsers)
    add_stack_parser(subparsers)
    add_fits_parser(subparsers)
    add_db_parser(subparsers)
    add_info_parser(subparsers)
    add_video_parser(subparsers)
    add_solve_parser(subparsers)
    add_organize_parser(subparsers)

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
