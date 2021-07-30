import json
import time
import os
import time
from typing import Tuple
import random
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

torch.set_printoptions(threshold=100000)


def padding_tokens(tokens, max_seq_length, pad_token, direct, max_context_length=0):
    if max_context_length == 0:
        max_context_length = max_seq_length

    if len(tokens) > max_context_length:
        if direct > 0:
            pad_tokens = tokens[:max_context_length]
        else:
            pad_tokens = tokens[-max_context_length:]
    else:
        pad_tokens = tokens
    token_len = len(pad_tokens)
    pad_tokens = pad_tokens + [pad_token for _ in range(max_seq_length - token_len)]
    return pad_tokens, token_len


def _reorder_cache(past: Tuple, beam_idx: Tensor) -> Tuple[Tensor]:
    return tuple(layer_past.index_select(1, beam_idx).contiguous().detach() for layer_past in past)


def _calc_banned_ngram_tokens(prev_input_ids: Tensor, num_hypos: int, no_repeat_ngram_size: int, cur_len: int) -> None:
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < no_repeat_ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]

    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

    def _get_generated_ngrams(hypo_idx):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
        return generated_ngrams[hypo_idx].get(ngram_idx, [])

    banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
    return banned_tokens


def _enforce_repetition_penalty_(lprobs, batch_size, num_beams, prev_output_tokens, repetition_penalty):
    """repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858). """

    for i in range(batch_size * num_beams):
        print('prev_output_tokens.shape', prev_output_tokens.shape)
        print('prev_output_tokens[i].shape', prev_output_tokens[i].shape)

        for previous_token in set(prev_output_tokens[i].tolist()):
            # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
            if lprobs[i, previous_token] < 0:
                lprobs[i, previous_token] *= repetition_penalty
            else:
                lprobs[i, previous_token] /= repetition_penalty


def _postprocess_next_token_scores(
        scores,
        history,
        cur_len,
        batch_size,
        num_beams,
        repetition_penalty=1.0,
        no_repeat_ngram_size=4,
        bad_words_ids=None,
        min_length=0,
        max_length=100,
        eos_token_id=None,
):
    # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
    if repetition_penalty != 1.0 and history is not None:
        _enforce_repetition_penalty_(scores, batch_size, num_beams, history, repetition_penalty)

    # score: batch_size * beam, vocab
    # set eos token prob to zero if min_length is not reached
    if eos_token_id is not None and cur_len < min_length:
        for eos in eos_token_id:
            scores[:, eos] = -float("inf")

    if no_repeat_ngram_size > 0 and history is not None:
        # calculate a list of banned tokens to prevent repetitively generating the same ngrams
        num_batch_hypotheses = batch_size * num_beams
        # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
        banned_batch_tokens = _calc_banned_ngram_tokens(
            history, num_batch_hypotheses, no_repeat_ngram_size, cur_len
        )

        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float("inf")

    return scores


def _add_beam_candidate(best_score, best_sequence, batch_size, num_beams, beam_scores, history, length_penalty, eos_token_id=None):
    last_tokens = history[:, -1]
    for _i in range(batch_size * num_beams):
        if eos_token_id is None or last_tokens[_i] in eos_token_id:
            cur_len = history.shape[-1]
            _score = beam_scores.view(-1)[_i] / cur_len ** length_penalty

            batch_id = _i // num_beams

            if not batch_id in best_score or best_score[batch_id] < _score:
                best_score[batch_id] = _score
                best_sequence[batch_id][:cur_len] = history[_i]

            beam_scores.view(-1)[_i] = -float("inf")


def beam2(model, data_iter, args):
    model.eval()
    total_loss = 0.
    start_time = time.time()

    all_predictions = {}
    with torch.no_grad():
        for idx, data in enumerate(data_iter):
            data = {key: value for key, value in data.items()}

            _id = data['id'].to(args.device)
            _query = data['query'].to(args.device)
            _query_len = data['query_len'].to(args.device)

            ## local adaptation start.

            ## local adaptation end.

            output = None
            score = None

            batch_size = _id.size(0)
            num_beams = args.beam
            length_penalty = args.length_penalty

            _batch = torch.arange(0, _id.size(0), device=args.device, dtype=torch.long)

            past = None
            len_past = None

            _query = _query.repeat(1, num_beams).view(batch_size * num_beams, -1)
            _query_len = _query_len.unsqueeze(-1).repeat(1, num_beams).view(-1)

            # scores for each sentence in the beam
            beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=_query.device)

            # for greedy decoding it is made sure that only tokens of the first beam are considered to avoid sampling the exact same tokens three times
            # beam_scores[:, 1:] = -1e9
            # beam_scores = beam_scores.view(-1)    # shape (batch_size * num_beams,)

            # self.beam_scores = []
            # beam_tokens = []
            # beam_idxes = []

            best_sequence = torch.zeros((batch_size, args.eval_len), dtype=torch.long, device=_query.device)
            best_score = {}

            history = None
            with torch.no_grad():
                for i in range(0, args.eval_len):
                    if i == 0:
                        logits, past = model(_query)
                        logits = logits[_batch, (_query_len - 1).long(), :]  # batch_size * beam, vocab
                    else:
                        # print('token_id.shape', token_id.shape, token_id)
                        # print('past.shape', past[0].shape)
                        # print('len_past.shape', len_past.shape, len_past)

                        logits, past = model(token_id, past=past, len_past=len_past)
                        logits = logits[:, -1, :]  # batch_size * beam, vocab

                    logits = _postprocess_next_token_scores(
                        logits,
                        history,
                        i,
                        batch_size,
                        num_beams,
                        repetition_penalty=args.repetition_penalty,
                        no_repeat_ngram_size=args.no_repeat_ngram_size,
                        min_length=args.min_length,
                        eos_token_id=args.eos_token_id,
                    )

                    softmax_probs = F.softmax(logits, dim=-1)
                    ##_prob, _w_idx = torch.topk(softmax_probs, num_beams) # batch_size, beam

                    vocab_size = softmax_probs.shape[-1]

                    _logprob = torch.log(softmax_probs)  # batch_size * beam, vocab
                    if i == 0:
                        next_scores = _logprob.view(batch_size, num_beams, -1)[:, 0, :]  # batch_size, vocab

                    else:
                        next_scores = beam_scores.unsqueeze(-1) + _logprob.view(batch_size, num_beams, -1)
                        next_scores = next_scores.view(batch_size, -1)  # batch_size, beam * vocab

                    # else:
                    #    next_scores = _logprob + beam_scores[:, None].expand_as(_logprob)    # (batch_size * num_beams, vocab_size)

                    # re-organize to group the beam together (we are keeping top hypothesis accross beams)
                    # next_scores = next_scores.view(
                    #        batch_size, num_beams * vocab_size
                    # )    # (batch_size, num_beams * vocab_size)

                    # print('vocab_size', vocab_size)
                    # print('next_scores.shape (1)', next_scores.shape)

                    next_scores, next_tokens = torch.topk(next_scores, num_beams, dim=1, largest=True,
                                                          sorted=True)  # batch_size, num_beams

                    # print('next_scores.shape (2)', next_scores.shape, next_scores)
                    # print('next_tokens.shape (2)', next_tokens.shape, next_tokens)

                    beam_id = (next_tokens // vocab_size).view(-1)  # batch_size * num_beams
                    token_id = (next_tokens % vocab_size).view(-1).unsqueeze(-1)  # batch_size, num_beams

                    beam_idx = beam_id.view(batch_size, num_beams) + (_batch * num_beams).unsqueeze(-1)
                    # past, 2, batch_size * beam, *, *, *,
                    # if past is not None:
                    # print('beam_id', beam_id)
                    # print('beam_idx', beam_idx)
                    # print('token_id', token_id.shape, token_id)

                    # print('past.shape (1)', past[0].shape)
                    past = _reorder_cache(past, beam_idx.view(-1))
                    # print('past.shape (2)', past[0].shape)

                    beam_scores = next_scores  # batch_size, num_beams

                    len_past = (_query_len + i).long()

                    if history is None:
                        history = token_id.detach()
                    else:
                        history = torch.cat((history[beam_idx.view(-1)], token_id.detach()), dim=1).detach()
                    # print('history.shape (1)', history.shape)
                    _add_beam_candidate(best_score, best_sequence, batch_size, num_beams, beam_scores, history,
                                        args.length_penalty, eos_token_id=args.eos_token_id)

                _add_beam_candidate(best_score, best_sequence, batch_size, num_beams, beam_scores, history,
                                    args.length_penalty)

            with torch.no_grad():
                output = torch.stack([best_sequence])

            _id = _id.view(-1).cpu()
            output = output.view(-1, output.shape[-1]).cpu()
            # score = score.view(-1, score.shape[-1]).cpu()

            for _b in range(0, _id.shape[-1]):
                _i = int(_id[_b].item())
                all_predictions[_i] = {}
                all_predictions[_i]['id'] = _i
                all_predictions[_i]['predict'] = output[_b].tolist()
                # all_predictions[_i]['score'] = score[_b].tolist()

            if idx % 10 == 0:
                print('inference samples', idx)

    if 'output_file' in args and len(args.output_file) > 0:
        pred_file = os.path.join(args.work_dir, args.output_file)
        print('saving prediction file', pred_file)
        with open(pred_file, 'w') as writer:
            for _i in all_predictions:
                writer.write(json.dumps(all_predictions[_i]) + '\n')
    else:
        return all_predictions


class FT_Dataset_3(Dataset):
    def __init__(self, ft_samples, batch_size, max_seq_length,
                 max_eval_length=0, joint_lm=False, prefix_len=0, infix_len=0,
                 prefix_cursor=1000000, infix_cursor=2000000):
        self.ft_samples = ft_samples
        self.batch_size = batch_size
        self.num_examples = len(self.ft_samples)
        self.max_seq_length = max_seq_length
        self.max_eval_length = max_eval_length
        self.rng = random.Random(911)
        self.joint_lm = joint_lm

        self.num_batches = int((self.num_examples + self.batch_size - 1) / self.batch_size)

        self.prefix_len = prefix_len
        self.infix_len = infix_len
        self.prefix_cursor = prefix_cursor
        self.infix_cursor = infix_cursor

    def __len__(self):
        return self.num_batches * self.batch_size

    def __getitem__(self, item):
        if item >= self.num_examples:
            item = self.rng.randint(0, self.num_examples - 1)

        example = self.ft_samples[item]
        context = example[0]
        completion = example[1]

        pretokens = [i + self.prefix_cursor for i in range(0, self.prefix_len)]
        intokens = [i + self.infix_cursor for i in range(0, self.infix_len)]

        conditions = pretokens + context + intokens
        _input, _input_len = padding_tokens(conditions + completion, self.max_seq_length, 0, 1)

        pad_targets = [0 for i in range(0, self.prefix_len)] + context + [0 for i in
                                                                          range(0, self.infix_len)] + completion
        _target, _ = padding_tokens(pad_targets[1:], self.max_seq_length, 0, 1)

        if not self.joint_lm:
            _msk = [0.0] * (len(conditions) - 1) + [1.0] * (_input_len - len(conditions))
        else:
            _msk = [1.0] * (_input_len - 1)

        _msk, _ = padding_tokens(_msk, self.max_seq_length, 0.0, 1)

        output = {"id": torch.tensor(item, dtype=torch.long)}

        _query, _query_len = padding_tokens(
            conditions, self.max_seq_length, 0, -1,
            max_context_length=self.max_seq_length - self.max_eval_length
        )
        output["query"] = torch.tensor(_query, dtype=torch.long)
        output["query_len"] = torch.tensor(_query_len, dtype=torch.long)
        output["input"] = torch.tensor(_input, dtype=torch.long)
        output["target"] = torch.tensor(_target, dtype=torch.long)
        output["mask"] = torch.tensor(_msk, dtype=torch.float)

        return output
