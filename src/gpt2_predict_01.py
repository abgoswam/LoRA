#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import json
import numpy as np

import encoder

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

import numpy
import io
import sys
import threading
import math
import random

import json
import collections
from collections import Counter
from collections import OrderedDict
from progress.bar import Bar as Bar
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input', default=None, type=str, help='ft input file')
parser.add_argument('--vocab', type=str, default=None, help='vocab path')
parser.add_argument('--add_bos', action='store_true', help='')
parser.add_argument('--add_eos', action='store_true', help='')
args = parser.parse_args()

if __name__ == "__main__":
    enc = encoder.get_encoder(args.vocab)

    ft_encoded_jsonls = []
    with open(args.input, 'r', encoding="utf8") as reader:
        line_idx = 0
        for line in tqdm(reader):
            items = json.loads(line.strip())
            context = items['context']
            completion = items['completion']

            bos = 50256
            eos = 50256
            context_bpes, _ = enc.encode(context)
            context_bpes += [bos] if args.add_bos else []

            completion_bpes, _ = enc.encode(' ' + completion)
            completion_bpes += [eos] if args.add_eos else []

            ft_encoded_json = {'context': context_bpes, 'completion': completion_bpes}

            ft_encoded_jsonls.append(ft_encoded_json)
            line_idx += 1


print("done")