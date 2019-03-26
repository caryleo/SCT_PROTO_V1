# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
FILENAME:       eval.py
BY:             Gary 2019.3.12
LAST MODIFIED:  2019.3.12
DESCRIPTION:    eval core file
"""

import logging

from six.moves import cPickle

import models
from tool.dataloader import *
from tool.dataloaderraw import *
import eval_utils
import tool.utils as utils
import torch


def evaluation(opts, device):
    # Load infos
    logging.info("Path to info: %s" % opts.info_path)
    assert opts.info_path != '', "Info_path must be specified."
    with open(opts.info_path, 'rb') as info_file:
        info = cPickle.load(info_file)

    opts.update({'split': 'test'})

    # override and collect parameters
    if len(opts.input_fc_dir) == 0:
        # opts.input_fc_dir = info['opts'].input_fc_dir
        # opts.input_att_dir = info['opts'].input_att_dir
        # opts.input_box_dir = getattr(info['opts'], 'input_box_dir', '')

        opts.input_captions_h5 = info['opts'].input_captions_h5
        opts.input_features_directory = info['opts'].input_features_directory
    if len(opts.input_json) == 0:
        opts.input_json = info['opts'].input_json
    if opts.batch_size == 0:
        opts.batch_size = info['opts'].batch_size
    if len(opts.train_id) == 0:
        opts.train_id = info['opts'].train_id

    ignore = ["train_id", "batch_size", "beam_size", "start_from", "language_eval"]
     
    for k in vars(info['opts']).keys():
        if k not in ignore:
            if k in vars(opts):
                assert vars(opts)[k] == vars(info['opts'])[k], k + ' option not consistent'
            else:
                vars(opts).update({k: vars(info['opts'])[k]})  # copy over options from model

    vocabulary = info['vocabulary']  # ix -> word mapping

    # Setup the model
    model = models.setup(opts)
    model.load_state_dict(torch.load(opts.model_path))
    model.to(device=device)
    model.eval()
    criterion = utils.LanguageModelCriterion()

    # Create the Data Loader instance
    if len(opts.image_folder) == 0:
        loader = DataLoader(opts)
    else:
        loader = DataLoaderRaw({'folder_path': opts.image_folder,
                                'coco_json': opts.coco_json,
                                'batch_size': opts.batch_size,
                                'cnn_model': opts.cnn_model})

    # When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
    # So make sure to use the vocab in infos file.
    loader.index_to_word = info['vocabulary']

    # Set sample options
    loss, split_predictions, lang_stats = eval_utils.eval_split(model, criterion, loader,
                                                                vars(opts))

    logging.info('loss: ', loss)
    if lang_stats is not None:
        logging.info(lang_stats)

    if opts.dump_json == 1:
        # dump the json
        json.dump(split_predictions, open('vis/vis.json', 'w'))
