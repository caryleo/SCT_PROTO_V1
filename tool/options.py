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

    # cuda device
    parser.add_argument('-cuda', "--cuda_device",
                        default=0,
                        help="The cuda device")

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

    # training
    parser.add_argument('-injson', "-input_json",
                        type=str,
                        default="data/sct_caps2idx.json",
                        help="path to the input json containing the image info and vocabulary")

    parser.add_argument('-infeatdir', "--input_features_directory",
                        type=str,
                        default="data/features",
                        help="directory of h5 files containing fc features and att features")

    parser.add_argument('-incaph5', "--input_captions_h5",
                        type=str,
                        default="data/sct_caps.h5",
                        help="path to the input h5 containing the indexed captions and index")

    parser.add_argument('-start', '--start_from',
                        type=str,
                        default=None,
                        help="""continue training from saved model at this path. Path must contain files saved by previous training process: 
                            'infos.pkl'         : configuration;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)

    parser.add_argument('-batch', "--batch_size",
                        default=64,
                        type=int,
                        help="size of one mini-batch")

    parser.add_argument('-capsperimg', "--captions_per_image",
                        default=5,
                        type=int,
                        help="number of captions to sample for each image during training")

    parser.add_argument('-beam', "--beam_size",
                        default=1,
                        type=int,
                        help="beam search size")

    parser.add_argument('-epo', "--epoch_num",
                        default=25,
                        type=int,
                        help="number of epoch")

    parser.add_argument('-dropout', "--dropout_prob",
                        default=0.5,
                        type=float,
                        help="prob value for dropout")

    parser.add_argument('-chkpt', "--checkpoint_path",
                        type=str,
                        default='save',
                        help='directory to store checkpointed models')

    parser.add_argument('-only', "--train_only",
                        type=int,
                        default=0,
                        help='if 1 then use 80k, else use 110k, including restval')

    opts = parser.parse_args()

    return opts
