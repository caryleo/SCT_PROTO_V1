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
    parser.add_argument('-id', "--train_id",
                        type=str,
                        default='',
                        help='an id identifying this run/job. used in cross-val and appended when writing progress '
                             'files')

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
                        help="probability value for dropout")

    parser.add_argument('-only', "--train_only",
                        type=int,
                        default=0,
                        help='if 1 then use 80k, else use 110k, including restval')

    parser.add_argument('-laneval', '--language_eval',
                        type=int,
                        default=0,
                        help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires '
                             'coco-caption code from Github.')

    # model
    parser.add_argument('-model', '--caption_model',
                        type=str,
                        default="fc",
                        choices=["fc, att2in, tdbu, elka"],
                        help="model we want to use")

    parser.add_argument('-rnn', '--rnn_size',
                        type=int,
                        default=512,
                        help='size of the rnn in number of hidden nodes in each layer')

    parser.add_argument('-layers', '--num_layers',
                        type=int,
                        default=1,
                        help='number of layers in the RNN')
    parser.add_argument('-type', '--rnn_type',
                        type=str,
                        default='lstm',
                        choices=['rnn', 'gru', 'lstm'],
                        help='the type of RNN')

    parser.add_argument('-inencsize', '--input_encoding_size',
                        type=int,
                        default=512,
                        help='the encoding size of each token in the vocabulary, and the image.')

    parser.add_argument('-atthidsize', '--att_hid_size',
                        type=int,
                        default=512,
                        help='the hidden size of the attention MLP; only useful in show_attend_tell; 0 if not using '
                             'hidden layer')

    parser.add_argument('-fcfeatsize', '--fc_feat_size',
                        type=int,
                        default=2048,
                        help='2048 for resnet, 4096 for vgg')

    parser.add_argument('-attfeatsize', '--att_feat_size',
                        type=int,
                        default=2048,
                        help='2048 for resnet, 512 for vgg')

    # optimization
    parser.add_argument('-opt', '--optim',
                        type=str,
                        default='adam',
                        choices=["adam", "rmsprop", "sgd", "sgdmom", "adagrad", "adam"],
                        help='optimization type')

    parser.add_argument('-lr', '--learning_rate',
                        type=float,
                        default=4e-4,
                        help='learning rate')

    parser.add_argument('-lrdecayst', '--learning_rate_decay_start',
                        type=int,
                        default=-1,
                        help='at which iteration to start decaying learning rate (-1 = dont) (in epoch)')

    parser.add_argument('-lrdecayeve', '--learning_rate_decay_every',
                        type=int,
                        default=3,
                        help='decay start from this epoch)')

    parser.add_argument('-lrdecay', '--learning_rate_decay_rate',
                        type=float,
                        default=0.8,
                        help='learning rate decay rate')

    parser.add_argument('-alpha', '--optim_alpha',
                        type=float,
                        default=0.9,
                        help='alpha for adam')

    parser.add_argument('-beta', '--optim_beta',
                        type=float,
                        default=0.999,
                        help='beta for adam')

    parser.add_argument('-epsilon', '--optim_epsilon',
                        type=float,
                        default=1e-8,
                        help='epsilon that goes into denominator for smoothing')

    parser.add_argument('-wdecay', '--weight_decay',
                        type=float,
                        default=0,
                        help='weight_decay')

    parser.add_argument('-ssst', '--scheduled_sampling_start',
                        type=int,
                        default=-1,
                        help='at which epoch to start decay gt probability')

    parser.add_argument('-ssinevery', '--scheduled_sampling_increase_every',
                        type=int,
                        default=5,
                        help='every how many iterations thereafter to gt probability')

    parser.add_argument('-ssincp', '--scheduled_sampling_increase_prob',
                        type=float,
                        default=0.05,
                        help='value to update the prob')

    parser.add_argument('-ssmaxp', '--scheduled_sampling_max_prob',
                        type=float,
                        default=0.25,
                        help='Maximum scheduled sampling prob.')

    parser.add_argument('-gclip', '--grad_clip',
                        type=float,
                        default=0.1,
                        help='clip gradients at this value')

    # check point
    parser.add_argument('-start', '--start_from',
                        type=str,
                        default=None,
                        help="""continue training from saved model at this path. Path must contain files saved by 
                        previous training process: 
                            'infos.pkl'     : configuration;
                            'checkpoint'    : paths to model file(s) (created by tf).
                             Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'  : file(s) with model definition (created by tf) """)

    parser.add_argument('-chkpt', "--checkpoint_path",
                        type=str,
                        default='checkpoints',
                        help='directory to store checkpointed models')

    parser.add_argument('-best', '--load_best_score',
                        type=int,
                        default=1,
                        help='Do we load previous best score when resuming training.')

    parser.add_argument('-losslogeve', '--losses_log_every',
                        type=int,
                        default=25,
                        help='How often do we snapshot losses, for inclusion in the progress dump (0 = disable)')

    parser.add_argument('-savechkpteve', '--save_checkpoint_every',
                        type=int,
                        default=2500,
                        help='how often to save a model checkpoint (in iterations)')

    opts = parser.parse_args()

    return opts
