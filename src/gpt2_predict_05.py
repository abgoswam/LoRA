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

parser.add_argument('--vocab', type=str, default='../vocab', help='vocab path')
parser.add_argument('--init_checkpoint', default=r'../trained_models/model.84000.pt', type=str,
                    help='initial checkpoint.')

parser.add_argument("--platform", default='local', type=str, help='run locally')
parser.add_argument("--random_seed", default=10, type=int, help='random seed')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--seq_len', type=int, default=512, help='number of tokens to predict.')
parser.add_argument('--eval_len', type=int, default=64, help='evaluation length.')
parser.add_argument('--min_length', type=int, default=0, help='minimum generation length.')
parser.add_argument('--model_card', default='gpt2.md', choices=['gpt2.sm', 'gpt2.md', 'gpt2.lg'], help='model names.')
parser.add_argument('--lora_dim', type=int, default=4, help='lora attn dimension.')
parser.add_argument('--lora_alpha', type=int, default=32, help='lora attn alpha.')
parser.add_argument('--beam', type=int, default=10, help='beam search size.')
parser.add_argument('--length_penalty', type=float, default=0.9, help='length penalty')
parser.add_argument('--no_repeat_ngram_size', type=int, default=4, help='no_repeat_ngram_size')
parser.add_argument('--repetition_penalty', type=float, default=1.0, help='repetition_penalty')
parser.add_argument('--eos_token_id', action='append', type=int, default=[50256, 628], help='eos token id')
parser.add_argument('--prefix_len', default=0, type=int, help='prefix length.')
parser.add_argument('--infix_len', default=0, type=int, help='infix length.')


def print_args(args):
    print('=' * 100)
    for k, v in args.__dict__.items():
        print('        - {} : {}'.format(k, v))
    print('=' * 100)


def initialize():
    args = parser.parse_args()

    # initialize : device to use cuda
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    args.device = device
    print_args(args)

    # initialize : encoder
    enc = encoder.get_encoder(args.vocab)

    # initialize model
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

    lm_net = lm_net.to(device)

    return args, device, enc, lm_net


if __name__ == "__main__":
    # initialize args, device, enc
    args, device, enc, model = initialize()

    ft_samples = []  # this is what we use to construct FT_Dataset
    contexts = [
        "In mathematics , the symbol × has a number of uses , including.The arrangement of the dots was variable and not necessarily linear . Five dots arranged like ( ⁙ ) ( as on the face of a die ) are known as a quincunx , from the name of the Roman fraction / coin . The Latin words sextans and quadrans are the source of the English words sextant and quadrant ..The arrangement of the dots was variable and not necessarily linear . Five dots arranged like ( ⁙ ) ( as on the face of a die ) are known as a quincunx , from the name of the Roman fraction / coin . The Latin words sextans and quadrans are the source of the English words sextant and quadrant .\n\nQ: what do the 3 dots mean in math?\n\nA: ",
        "`` Photograph '' is a song by English musician Ringo Starr that was released as the lead single from his 1973 album Ringo . Starr co-wrote the song with George Harrison , his former bandmate from the Beatles . Although the two of them collaborated on other compositions , it is the only song officially credited to the pair . A signature tune for Starr as a solo artist , `` Photograph '' became an international hit , topping singles charts in the United States , Canada and Australia , and receiving gold disc certification for US sales of 1 million . Music critics have similarly received the song favourably ; Stephen Thomas Erlewine of AllMusic considers it to be `` among the very best post-Beatles songs by any of the Fab Four '' ..`` Photograph '' has appeared on Starr 's compilation albums Blast from Your Past ( 1975 ) and Photograph : The Very Best of Ringo Starr ( 2007 ) , and live versions have featured on releases recorded with his All - Starr Band and with the Roundheads . In November 2002 , a year after Harrison 's death , Starr sang `` Photograph '' at the Concert for George -- a performance that was an emotional highpoint of the event . Engelbert Humperdinck , Camper Van Beethoven , Cilla Black and Adam Sandler are among the artists who have covered the song ..The song provided the title to Starr 's 2007 career - spanning compilation Photograph : The Very Best of Ringo Starr , having earlier appeared on Blast from Your Past ( 1975 ) , a greatest - hits collection covering his years on Apple Records . For the 1991 CD reissue of Ringo , the album was expanded through the inclusion of three bonus tracks , the last of which was the long - unavailable `` Down and Out '' . In 2009 , `` Photograph '' was featured in the Judd Apatow - directed film Funny People and appeared on the accompanying soundtrack album .\n\nQ: who wrote the song photograph by ringo starr?\n\nA: ",
        "Richards has become a successful artist . Her paintings , usually of landscapes and figures , tend to be in the style of the impressionists . In October 2005 , she won first place in the National Professional Oil Painting Competition ( sponsored by American Artist magazine ) for the painting Lady of the Dahlias ..A list of paintings by Abanindranath Tagore :.Jennings said the people who saved the painting and removed the objects actually were :\n\nQ: where does the last name painter come from?\n\nA: "
    ]

    line_idx = 0
    ft_encoded_jsonls = []
    for context in contexts:
        completion = ''

        bos = 50256
        eos = 50256
        context_bpes, _ = enc.encode(context)
        context_bpes += [bos]

        completion_bpes, _ = enc.encode(' ' + completion)
        completion_bpes += [eos]

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

    print('model sampling ...')
    all_predictions = beam2(model, valid_loader, args)

    sampled = []
    for i in range(len(all_predictions)):
        _dict = all_predictions[i]
        _id = _dict['id']
        _pred_tokens = _dict['predict']
        sample = enc.decode(_pred_tokens).split('<|endoftext|>')[0].split('\n\n')[0].strip()
        sampled.append(sample)

    for item in sampled:
        print(item)

print("done")
