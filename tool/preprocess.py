# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
FILENAME: preprocess.py
BY: Gary 2019.3.14
LAST MODIFIED: 2019.3.14
DESCRIPTION: preprocess core
"""


def preprocess_captions(args):
    path_to_input_json = args.input_caption_json
    path_to_ouput_json = args.output_caption_json
    path_to_output_h5 = args.output_caption_h5

    assert path_to_input_json != "", "Path to input json is needed."

    print("The path to input caption json:", path_to_input_json)
    print("The path to output caption json:", path_to_ouput_json)
    print("The path to output caption h5:", path_to_output_h5)
