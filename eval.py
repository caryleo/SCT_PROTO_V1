# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
FILENAME:       eval.py
BY:             Gary 2019.3.12
LAST MODIFIED:  2019.3.12
DESCRIPTION:    eval core file
"""

import json
import numpy as np

from six.moves import cPickle

import models
from tool.dataloader import *
from tool.dataloaderraw import *
import eval_utils
import tool.utils as utils
import torch


def eval(opts, device):
    # Load infos
    with open(opts.infos_path) as f:
        infos = cPickle.load(f)

    # override and collect parameters
    if len(opts.input_fc_dir) == 0:
        opts.input_fc_dir = infos['opts'].input_fc_dir
        opts.input_att_dir = infos['opts'].input_att_dir
        opts.input_box_dir = getattr(infos['opts'], 'input_box_dir', '')
        opts.input_label_h5 = infos['opts'].input_label_h5
    if len(opts.input_json) == 0:
        opts.input_json = infos['opts'].input_json
    if opts.batch_size == 0:
        opts.batch_size = infos['opts'].batch_size
    if len(opts.id) == 0:
        opts.id = infos['opts'].id
    ignore = ["id", "batch_size", "beam_size", "start_from", "language_eval"]
    for k in vars(infos['opts']).keys():
        if k not in ignore:
            if k in vars(opts):
                assert vars(opts)[k] == vars(infos['opts'])[k], k + ' option not consistent'
            else:
                vars(opts).update({k: vars(infos['opts'])[k]})  # copy over options from model

    vocab = infos['vocab']  # ix -> word mapping

    # Setup the model
    model = models.setup(opts)
    model.load_state_dict(torch.load(opts.model))
    model.cuda()
    model.eval()
    crit = utils.LanguageModelCriterion()

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
    loader.ix_to_word = infos['vocab']

    # Set sample options
    loss, split_predictions, lang_stats = eval_utils.eval_split(model, crit, loader,
                                                                vars(opts))

    print('loss: ', loss)
    if lang_stats:
        print(lang_stats)

    if opts.dump_json == 1:
        # dump the json
        json.dump(split_predictions, open('vis/vis.json', 'w'))
