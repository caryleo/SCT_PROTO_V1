# !/usr/bin/env python3
# encoding: utf-8

# FILENAME: sct-proto-v1.py
# BY: Gary 2019.3.12
# LAST MODIFIED: 2019.3.12
# DESCRIPTION: main core

import options

# arguments
opts = options.tool_parse_arg()

# mode
if opts.mode:
    print("Training")
else:
    print("Evaluating")


