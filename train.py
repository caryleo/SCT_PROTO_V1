# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
FILENAME:       train.py
BY:             Gary 2019.3.12
LAST MODIFIED:  2019.3.12
DESCRIPTION:    train core file
"""

import logging

from tool.dataloader import DataLoader

try:
    import tensorflow as tf
except:
    logging.warning("Tensorflow is not installed, no Tensorboard logging.")
    tf = None


def add_summary_value(writer, key, value, iteration):
    summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    writer.add_summary(summary, iteration)


def train(opts):
    loader = DataLoader(opts)
    return
