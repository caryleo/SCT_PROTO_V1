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
                        choices=["train", "eval", "precaps", "prefeats"],
                        help="Choose the mode")

    parser.add_argument('-d', '--debug',
                        action='store_true',
                        help="Debug mode")

    # preprocess captions
    parser.add_argument('-incapjson', "--input_caption_json",
                        default="",
                        help="Path to input split json to preprocess the captions")

    parser.add_argument('-outcapjson', "--output_caption_json",
                        default="data/sct_caps2idx.json",
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
                        help="root directory of images")

    # preprocess images, including image_root and input_caption_json
    parser.add_argument('-outfeatdir', "--output_feature_directory",
                        default='data/features',
                        help="directory for output feature files")

    parser.add_argument('-attsize', "--attention_size",
                        default=14,
                        choices=[7, 14],
                        type=int,
                        help="attention size")

    parser.add_argument('-mod', "--model",
                        default="resnet101",
                        choices=["resnet50", "resnet101", "resnet152"],
                        type=str,
                        help="model for feature extraction")

    parser.add_argument('-moddir', "--model_directory",
                        default="data/models",
                        type=str,
                        help="directory of the models for feature extraction")

    opts = parser.parse_args()

    # validation

    return opts
