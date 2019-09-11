import torch
import numpy as np
import sys

from utils import vocab, pos_vocab, ner_vocab, rel_vocab

class Example:

    def __init__(self, input_dict):
        self.id = input_dict['id']
        self.passage = input_dict['d_words']
        self.question = input_dict['q_words']
        self.choice = input_dict['c_words']
        self.d_pos = input_dict['d_pos']
        self.d_ner = input_dict['d_ner']
        self.q_pos = input_dict['q_pos']
        self.valid = 0
        #if len(self.d_pos) == len(self.passage.split()):
        assert len(self.q_pos) == len(self.question.split()), (self.q_pos, self.question)
        assert len(self.d_pos) == len(self.passage.split())
        self.features = np.stack([input_dict['in_q'], input_dict['in_c'], \
                                    input_dict['lemma_in_q'], input_dict['lemma_in_c'], \
                                    input_dict['tf']], 1)
        assert len(self.features) == len(self.passage.split())
        self.label = input_dict['label']

        if len(self.passage.split()) == 0 or len(self.question.split()) == 0 or len(self.choice.split()) == 0:
            print(self.id)
        self.d_tensor = torch.LongTensor([vocab[w] for w in self.passage.split()])
        self.q_tensor = torch.LongTensor([vocab[w] for w in self.question.split()])
        self.c_tensor = torch.LongTensor([vocab[w] for w in self.choice.split()])
        self.d_pos_tensor = torch.LongTensor([pos_vocab[w] for w in self.d_pos])
        self.q_pos_tensor = torch.LongTensor([pos_vocab[w] for w in self.q_pos])
        self.d_ner_tensor = torch.LongTensor([ner_vocab[w] for w in self.d_ner])
        self.features = torch.from_numpy(self.features).type(torch.FloatTensor)
        self.p_q_relation = torch.LongTensor([rel_vocab[r] for r in input_dict['p_q_relation']])
        self.p_c_relation = torch.LongTensor([rel_vocab[r] for r in input_dict['p_c_relation']])
        self.valid = 1

    def __str__(self):
        return 'Passage: %s\n Question: %s\n Answer: %s, Label: %d' % (self.passage, self.question, self.choice, self.label)

def _to_indices_and_mask(batch_tensor, need_mask=True):
    mx_len = max([t.size(0) for t in batch_tensor])
    batch_size = len(batch_tensor)
    indices = torch.LongTensor(batch_size, mx_len).fill_(0)
    if need_mask:
        mask = torch.ByteTensor(batch_size, mx_len).fill_(1)
    for i, t in enumerate(batch_tensor):
        indices[i, :len(t)].copy_(t)
        if need_mask:
            mask[i, :len(t)].fill_(0)
    if need_mask:
        return indices, mask
    else:
        return indices

def _to_feature_tensor(features):
    mx_len = max([f.size(0) for f in features])
    batch_size = len(features)
    f_dim = features[0].size(1)
    f_tensor = torch.FloatTensor(batch_size, mx_len, f_dim).fill_(0)
    for i, f in enumerate(features):
        f_tensor[i, :len(f), :].copy_(f)
    return f_tensor

def batchify(choice_data, nli_data):
    batch_data = []

    for i in range(len(nli_data)):
        c1_id, c2_id, nli_label = nli_data[i]
        batch_data.append((choice_data[c1_id], choice_data[c2_id], nli_label))

    p1, p1_mask = _to_indices_and_mask([ex[0].d_tensor for ex in batch_data])
    p1_pos = _to_indices_and_mask([ex[0].d_pos_tensor for ex in batch_data], need_mask=False)
    p1_ner = _to_indices_and_mask([ex[0].d_ner_tensor for ex in batch_data], need_mask=False)
    p1_q1_relation = _to_indices_and_mask([ex[0].p_q_relation for ex in batch_data], need_mask=False)
    p1_c1_relation = _to_indices_and_mask([ex[0].p_c_relation for ex in batch_data], need_mask=False)
    q1, q1_mask = _to_indices_and_mask([ex[0].q_tensor for ex in batch_data])
    q1_pos = _to_indices_and_mask([ex[0].q_pos_tensor for ex in batch_data], need_mask=False)
    c1, c1_mask = _to_indices_and_mask([ex[0].c_tensor for ex in batch_data])
    f1_tensor = _to_feature_tensor([ex[0].features for ex in batch_data])
    y1 = torch.FloatTensor([ex[0].label for ex in batch_data])

    p2, p2_mask = _to_indices_and_mask([ex[1].d_tensor for ex in batch_data])
    p2_pos = _to_indices_and_mask([ex[1].d_pos_tensor for ex in batch_data], need_mask=False)
    p2_ner = _to_indices_and_mask([ex[1].d_ner_tensor for ex in batch_data], need_mask=False)
    p2_q2_relation = _to_indices_and_mask([ex[1].p_q_relation for ex in batch_data], need_mask=False)
    p2_c2_relation = _to_indices_and_mask([ex[1].p_c_relation for ex in batch_data], need_mask=False)
    q2, q2_mask = _to_indices_and_mask([ex[1].q_tensor for ex in batch_data])
    q2_pos = _to_indices_and_mask([ex[1].q_pos_tensor for ex in batch_data], need_mask=False)
    c2, c2_mask = _to_indices_and_mask([ex[1].c_tensor for ex in batch_data])
    f2_tensor = _to_feature_tensor([ex[1].features for ex in batch_data])
    y2 = torch.FloatTensor([ex[1].label for ex in batch_data])

    nli_y = torch.LongTensor([tup[2] for tup in batch_data])

    return p1, p1_pos, p1_ner, p1_mask, q1, q1_pos, q1_mask, c1, c1_mask, f1_tensor, p1_q1_relation, p1_c1_relation, p2, p2_pos, p2_ner, p2_mask, q2, q2_pos, q2_mask, c2, c2_mask, f2_tensor, p2_q2_relation, p2_c2_relation, y1, y2, nli_y
