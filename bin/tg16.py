#!/usr/bin/env python3

from pygraso.preprocessing import tg16
import argparse

argparser = argparse.ArgumentParser(
    description="Run g16 and then delete log file and rwf file"
)

argparser.add_argument("input_file", type=str, help="g16 gjf path")
argparser.add_argument(
    "--deriv",
    "-d",
    type=bool,
    default=True,
    help="store derivative properties (default: True)",
)

argparser.add_argument(
    "--method",
    "-m",
    type=str,
    default="1",
    help="calculation method for orbital perturbation response matrix U. Please select 1 or 2 (default: '1')",
)

argparser.add_argument(
    "--work_dir",
    "-w",
    type=str,
    default=None,
    help="working directory where calculation is done (default: $GAUSS_SCRDIR)",
)
args = argparser.parse_args()

tg16(args.input_file, work_dir=args.work_dir, method=args.method, deriv=args.deriv)
