import argparse
import logging

import pandas as pd

from functions import *

pd.set_option('display.width', 1000)


def main():
    parser = argparse.ArgumentParser(description="Manipulate massive dataset files")
    generate_sub_parser = parser.add_argument_group(title="generate", description="Generate an en-xx mapping")
    generate_sub_parser.add_argument('-r', "--ref", help="English input file mapping,only single file",
                                     dest="input")
    generate_sub_parser.add_argument('-i', "--in",
                                     help="Other input files,can be specified more than once, mutually exclusive with "
                                          "--dir",
                                     dest="output", action="append")
    generate_sub_parser.add_argument("-d", "--dir",
                                     help="Input directory, this will read all files in the directory" +
                                          " and create a en-XX mapping for it. Mutually exclusive with --in",
                                     dest="dir")
    parser.add_argument("--log", help="Enable logging in the known levels", choices=["info", "trace", "debug"],
                        dest="log")

    args = parser.parse_args()

    if args.log:
        # Enable logging
        numeric_level = getattr(logging, args.log.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: %s' % args.log)
        logging.basicConfig(level=numeric_level)

    if args.dir and args.output:
        logging.error("Dir and output are mutually exclusive, choose one of them")
        return

    if args.input and args.output:
        generate_file(args.input, args.output)
    elif args.input and args.dir:
        walk_directory(args.input, args.dir)


if __name__ == "__main__":
    main()