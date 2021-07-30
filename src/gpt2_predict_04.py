#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import argparse
import json
import os
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from gpt2_predict_utils import beam2, FT_Dataset_3
import encoder
from model import GPT2Config, GPT2LMModel

torch.set_printoptions(threshold=100000)

parser = argparse.ArgumentParser()
parser.add_argument('--input', default=None, type=str, help='ft input file')
parser.add_argument('--vocab', type=str, default=None, help='vocab path')
parser.add_argument('--add_bos', action='store_true', help='')
parser.add_argument('--add_eos', action='store_true', help='')

parser.add_argument("--platform", default='local', type=str, help='run locally')
parser.add_argument("--random_seed", default=10, type=int, help='random seed')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--seq_len', type=int, default=512, help='number of tokens to predict.')
parser.add_argument('--eval_len', type=int, default=64, help='evaluation length.')
parser.add_argument('--min_length', type=int, default=0, help='minimum generation length.')
parser.add_argument('--model_card', default='gpt2.md', choices=['gpt2.sm', 'gpt2.md', 'gpt2.lg'], help='model names.')
parser.add_argument('--init_checkpoint', default=None, type=str, help='initial checkpoint.')
parser.add_argument('--lora_dim', type=int, default=4, help='lora attn dimension.')
parser.add_argument('--lora_alpha', type=int, default=32, help='lora attn alpha.')
parser.add_argument('--work_dir', type=str, default=os.getenv('PT_OUTPUT_DIR', 'gpt2_model'), help='working folder.')
parser.add_argument('--beam', type=int, default=10, help='beam search size.')
parser.add_argument('--length_penalty', type=float, default=0.9, help='length penalty')
parser.add_argument('--no_repeat_ngram_size', type=int, default=4, help='no_repeat_ngram_size')
parser.add_argument('--repetition_penalty', type=float, default=1.0, help='repetition_penalty')
parser.add_argument('--eos_token_id', action='append', type=int, default=[50256], help='eos token id')
parser.add_argument('--prefix_len', default=0, type=int, help='prefix length.')
parser.add_argument('--infix_len', default=0, type=int, help='infix length.')


def print_args(args):
    print('=' * 100)
    for k, v in args.__dict__.items():
        print('        - {} : {}'.format(k, v))
    print('=' * 100)


if __name__ == "__main__":
    args = parser.parse_args()

    # set device to use cuda
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    args.device = device
    print_args(args)

    enc = encoder.get_encoder(args.vocab)

    ft_samples = []  # this is what we use to construct FT_Dataset
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

            context = ft_encoded_json['context']
            completion = ft_encoded_json['completion']
            ft_samples.append([context, completion])
            line_idx += 1

    print(ft_encoded_jsonls)

    valid_data = FT_Dataset_3(ft_samples, args.batch_size, args.seq_len, args.eval_len, prefix_len=args.prefix_len,
                              infix_len=args.infix_len)
    valid_loader = DataLoader(valid_data,
                              sampler=SequentialSampler(valid_data),  # Pull out batches sequentially.
                              batch_size=args.batch_size, num_workers=0, shuffle=False, pin_memory=False,
                              drop_last=False)

    if args.model_card == 'gpt2.sm':
        config = GPT2Config(n_embd=768, n_layer=12, n_head=12, lora_attn_dim=args.lora_dim,
                            lora_attn_alpha=args.lora_alpha, prefix_len=args.prefix_len, infix_len=args.infix_len)
    elif args.model_card == 'gpt2.md':
        config = GPT2Config(n_embd=1024, n_layer=24, n_head=16, lora_attn_dim=args.lora_dim,
                            lora_attn_alpha=args.lora_alpha, prefix_len=args.prefix_len, infix_len=args.infix_len)
    elif args.model_card == 'gpt2.lg':
        config = GPT2Config(n_embd=1280, n_layer=36, n_head=20, lora_attn_dim=args.lora_dim,
                            lora_attn_alpha=args.lora_alpha, prefix_len=args.prefix_len, infix_len=args.infix_len)

    lm_net = GPT2LMModel(config)
    if args.init_checkpoint is not None:
        print('loading model pretrained weight.')
        cp = torch.load(args.init_checkpoint, map_location=torch.device('cpu'))
        lm_net.load_weight(cp)
    lm_net = lm_net.cuda()

    print('model sampling ...')
    all_predictions = beam2(lm_net, valid_loader, args)

    sampled = []
    for i in range(len(all_predictions)):
        _dict = all_predictions[i]
        _id = _dict['id']
        _pred_tokens = _dict['predict']
        sample = enc.decode(_pred_tokens).split('<|endoftext|>')[0].split('\n\n')[0].strip()
        sampled.append(sample)

    print(sampled)

print("done")
