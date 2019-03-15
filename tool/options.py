# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
FILENAME:       options.py
BY:             Gary 2019.3.12
LAST MODIFIED:  2019.3.15
DESCRIPTION:    reads arguments from the command line
"""

import argparse


def parse_arg_elka():
    parser = argparse.ArgumentParser(description="SCT PROTO V1 #ELKA")

    # core mode
    parser.add_argument('-m', "--mode",
                        dest="mode",
                        default="train",
                        required=True,
                        choices=["train", "eval", "precaps", "prefeat"],
                        help="Choose the mode")

    parser.add_argument('-d', '--debug',
                        action='store_true',
                        help="Debug mode")

    # preprocess captions
    parser.add_argument('-incapjson', "--input_caption_json",
                        default="",
                        help="Path to input split json to preprocess the captions")

    parser.add_argument('-outcapjson', "--output_caption_json",
                        default="data/sct_capindices.json",
                        help="Path to output the result json of processing the captions")

    parser.add_argument('-outcaph5', "--output_caption_h5",
                        default="data/sct_caps.h5",
                        help="Path to output the result h5 of processing the captions")

    parser.add_argument('-maxlen', "--max_sentence_length",
                        default=16,
                        help="The maximal length of sentence, exceeding part will be truncated")

    parser.add_argument('-wordthres', "--word_threshold",
                        default=5,
                        help="The threshold of the number of occurrences of words in all captions, "
                             "rare words will be replaced by token UNK")
    parser.add_argument('-imgrt', "--image_root",
                        default="",
                        help="root dictionary of images")

    opts = parser.parse_args()

    # validation

    return opts
