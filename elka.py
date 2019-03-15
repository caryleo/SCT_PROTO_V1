# !/usr/bin/env python3
# -*- coding: UTF-8 -*-


"""
FILENAME: elka.py
BY: Gary 2019.3.12
LAST MODIFIED: 2019.3.14
DESCRIPTION: main core file
"""
import json

from tool import options, preprocess


if __name__ == "__main__":
    # arguments
    opts = options.parse_arg_elka()

    # debug: opts
    print("Options input:")
    para = vars(opts)
    print(json.dumps(para, indent=2))

    # check mode
    print("Current core mode: ", end="")
    if opts.mode == 'train':
        print("Training")
    elif opts.mode == 'eval':
        print("Evaluating")
    elif opts.mode == 'precaps':
        print("Preprocessing captions")
        preprocess.preprocess_captions(opts)
    elif opts.mode == 'prefeat':
        print("Preprocessing features")
