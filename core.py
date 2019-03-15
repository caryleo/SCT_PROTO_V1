# !/usr/bin/env python3
# -*- coding: UTF-8 -*-


"""
FILENAME:       core.py
BY:             Gary 2019.3.12
LAST MODIFIED:  2019.3.15
DESCRIPTION:    main core file
"""
import json
import logging

from tool import options, preprocess


if __name__ == "__main__":
    # arguments
    opts = options.parse_arg_elka()

    # logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s  %(filename)s : %(levelname)s  %(message)s',
        datefmt='%Y.%m.%d-%H:%M:%S',
        filename='elka.log',
        filemode='w'
    )
    console = logging.StreamHandler()
    if opts.debug:
        console.setLevel(logging.DEBUG)
    else:
        console.setLevel(logging.INFO)
    console.setFormatter(
        logging.Formatter('%(asctime)s-%(levelname)s %(message)s',
                          datefmt='%Y.%m.%d-%H:%M:%S'))
    logging.getLogger('').addHandler(console)

    # DEBUG: opts
    para = vars(opts)
    logging.debug("Options input: \n" + json.dumps(para, indent=2))

    # check mode
    if opts.mode == 'train':
        logging.info("Current core mode: Training")
    elif opts.mode == 'eval':
        logging.info("Current core mode: Evaluating")
    elif opts.mode == 'precaps':
        logging.info("Current core mode: Preprocessing captions")
        preprocess.preprocess_captions(opts)
    elif opts.mode == 'prefeat':
        logging.info("Current core mode: Preprocessing features")
