import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from allennlp.data.dataset_readers.dataset_utils import enumerate_spans

from utils.entities import get_entities


class Example(object):
    def __init__(self, id, words, bio_labels, tokenizer, max_len):
        self.id = id
        self.words = words
        self.bio_labels = bio_labels
        self.max_len = max_len
        self.tokens, self.valid_mask, self.offsets = self._tokenize(tokenizer)
        self.entities = self._get_entities()

    def _tokenize(self, tokenizer):
        tokens, valid_mask, offsets = [], [], []

        for word in self.words:
            subwords = tokenizer.tokenize(word)
            offsets.append([len(tokens), len(tokens) + len(subwords) - 1])
            for i, word_token in enumerate(subwords):
                valid_mask.append(int(i == 0))
                tokens.append(word_token)
        return tokens, valid_mask, offsets

    def _get_entities(self):
        bio_labels = self.bio_labels
        max_len = self.max_len
        entities = get_entities(bio_labels)

        for _, start, end in entities:
            if start <= max_len - 1 and end > max_len - 1:
                bio_labels = bio_labels[:start-1] + ['O'] * (max_len - start)

        self.tokens = self.tokens[:max_len]
        self.valid_mask = self.valid_mask[:max_len]

        return get_entities(bio_labels)


def load_examples(args, mode, tokenizer):
    data_path = args.data_path
    max_len = args.max_seq_len
    file_path = os.path.join(data_path, '{}.txt'.format(mode))

    examples = []
    id = 1

    with open(file_path, encoding='utf-8') as f:
        words = []
        bio_labels = []

        for line in f:
            if line.startswith('-DOCSTART-') or line == '' or line == '\n':
                if words:
                    examples.append(
                        Example(
                            id='{}-{}'.format(mode, id),
                            words=words,
                            bio_labels=bio_labels,
                            tokenizer=tokenizer,
                            max_len=max_len
                        )
                    )
                    id += 1
                    words = []
                    bio_labels = []
            else:
                splits = line.split()
                words.append(splits[0])
                bio_labels.append(splits[-1].replace('\n', ''))
        if words:
            examples.append(
                Example(
                    id='{}-{}'.format(mode, id),
                    words=words,
                    bio_labels=bio_labels,
                    tokenizer=tokenizer,
                    max_len=max_len
                )
            )
    return examples

class SpanNERDataset(Dataset):
    def __init__(self, examples, args, tokenizer):
        self.examples = examples
        self.args = args
        self.features = self.get_features(examples, tokenizer)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index], self.features[index]

    def get_features(self, examples, tokenzier):
        labels = self.args.labels
        max_seq_len = self.args.max_seq_len
        max_span_len = self.args.max_span_len
        entity_weight, non_entity_weight = 1.0, 0.5
        pad_token_segment_id = 0
        pad_label_id = 0

        features = []
        label_map = {entity_type: idx for idx, entity_type in enumerate(labels)}

        for idx, example in enumerate(examples):
            tokens = example.tokens
            entity_span_idxes = [entity[1:] for entity in example.entities]
            span_word_idxes = enumerate_spans(example.words, offset=0, max_span_width=max_span_len)

            special_token_counts = 2
            max_token_num = max_seq_len - special_token_counts

            span_token_idxes, n_keep = word2token(example.offsets, span_word_idxes, max_token_num)

            span_weights = []
            span_labels = []

            for i, span_idx in enumerate(span_word_idxes[: n_keep]):
                if span_idx in entity_span_idxes:
                    span_weights.append(entity_weight)
                    span_type = example.entities[entity_span_idxes.index(span_idx)][0]
                    span_labels.append(label_map[span_type])
                else:
                    span_weights.append(non_entity_weight)
                    span_labels.append(0)

            span_lens = [idxes[1] - idxes[0] + 1 for idxes in span_word_idxes[: n_keep]]
            span_masks = np.ones_like(span_labels).tolist()

            max_num_span = max_seq_len * max_span_len - int((max_span_len - 1) * max_span_len / 2)

            span_masks = span_padding(span_masks, 0, max_num_span)
            span_labels = span_padding(span_labels, pad_label_id, max_num_span)
            span_lens = span_padding(span_lens, 0, max_num_span)
            span_weights = span_padding(span_weights, 0, max_num_span)
            span_token_idxes = span_padding(span_token_idxes, (0,0), max_num_span)
            span_word_idxes = span_padding(span_word_idxes, (0,0), max_num_span)

            tokens = tokens[: (max_seq_len - special_token_counts)]
            tokens = [tokenzier.cls_token] + tokens + [tokenzier.sep_token]

            pad_len = max_seq_len - len(tokens)
            input_ids = tokenzier.convert_tokens_to_ids(tokens + [tokenzier.pad_token] * pad_len)
            input_mask = [1] * len(tokens) + [0] * pad_len
            segment_ids = [0] * len(tokens) + [pad_token_segment_id] * pad_len

            features.append([
                input_ids, input_mask, segment_ids, span_token_idxes, span_labels,
                span_weights, span_lens, span_masks, span_word_idxes
            ])
        return features

def collate_fn(batch):
    """
    将数据集的batch样本变成可以输入模型的tensor
    batch: (examples, features)
    """

    features = []
    features.append({
        'input_ids': torch.LongTensor([sample[1][0] for sample in batch]),
        'input_mask': torch.LongTensor([sample[1][1] for sample in batch]),
        'segment_ids': torch.LongTensor([sample[1][2] for sample in batch]),
        'span_token_idxes': torch.LongTensor([sample[1][3] for sample in batch]),
        'span_labels': torch.LongTensor([sample[1][4] for sample in batch]),
        'span_weights': torch.Tensor([sample[1][5] for sample in batch]),
        'span_lens': torch.LongTensor([sample[1][6] for sample in batch]),
        'span_masks': torch.LongTensor([sample[1][7] for sample in batch]),
        'span_word_idxes': torch.LongTensor([sample[1][8] for sample in batch])
    })
    features.append({
        'words': [sample[0].words for sample in batch],
        'bio_labels': [sample[0].bio_labels for sample in batch]
    })

    return features


def word2token(offsets, span_idxes, max_len):
    """
    将span_idxes从word-level变为token-level
    """
    n_span_keep = 0
    span_idxes_ltoken = []

    for start, end in span_idxes:
        if offsets[end][-1] > max_len - 1:
            continue
        n_span_keep += 1
        span_idxes_ltoken.append((offsets[start][0] + 1, offsets[end][-1] + 1))

    return span_idxes_ltoken, n_span_keep

def span_padding(list, value, max_len):
    while len(list) < max_len:
        list.append(value)
    return list[:max_len]