# !/usr/bin/env python3
# encoding: utf-8

# FILENAME: options.py
# BY: Gary 2019.3.12
# LAST MODIFIED: 2019.3.12
# DESCRIPTION: reads arguments from the command line

import argparse


def tool_parse_arg():
    parser = argparse.ArgumentParser(description="Read and parse the arguments")

    parser.add_argument('-train', "--mode_train",
                        action="store_true",
                        dest="mode",
                        default=True,
                        help="Training mode")
    parser.add_argument('-eval', "--mode_eval",
                        action="store_false",
                        dest="mode",
                        help="Evaluation mode")

    ans = parser.parse_args()

    # validation

    return ans
