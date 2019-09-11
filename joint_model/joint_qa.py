import torch
import torch.nn as nn
import torch.nn.functional as F

import layers
from utils import vocab, pos_vocab, ner_vocab, rel_vocab

class JointQA(nn.Module):

    def __init__(self, args):
        super(JointQA, self).__init__()
        self.args = args
        self.embedding_dim = 300
        self.embedding = nn.Embedding(len(vocab), self.embedding_dim, padding_idx=0)
        self.embedding.weight.data.fill_(0)
        self.embedding.weight.data[:2].normal_(0, 0.1)
        self.pos_embedding = nn.Embedding(len(pos_vocab), args.pos_emb_dim, padding_idx=0)
        self.pos_embedding.weight.data.normal_(0, 0.1)
        self.ner_embedding = nn.Embedding(len(ner_vocab), args.ner_emb_dim, padding_idx=0)
        self.ner_embedding.weight.data.normal_(0, 0.1)
        self.rel_embedding = nn.Embedding(len(rel_vocab), args.rel_emb_dim, padding_idx=0)
        self.rel_embedding.weight.data.normal_(0, 0.1)
        self.RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU}

        self.p_q_emb_match = layers.SeqAttnMatch(self.embedding_dim)
        self.c_q_emb_match = layers.SeqAttnMatch(self.embedding_dim)
        self.c_p_emb_match = layers.SeqAttnMatch(self.embedding_dim)

        # Input size to RNN: word emb + question emb + pos emb + ner emb + manual features
        doc_input_size = 2 * self.embedding_dim + args.pos_emb_dim + args.ner_emb_dim + 5 + 2 * args.rel_emb_dim

        # RNN document encoder
        self.doc_rnn = layers.StackedBRNN(
            input_size=doc_input_size,
            hidden_size=args.hidden_size,
            num_layers=args.doc_layers,
            dropout_rate=0,
            dropout_output=args.dropout_rnn_output,
            concat_layers=False,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding)

        # RNN question encoder: word emb + pos emb
        qst_input_size = self.embedding_dim + args.pos_emb_dim
        self.question_rnn = layers.StackedBRNN(
            input_size=qst_input_size,
            hidden_size=args.hidden_size,
            num_layers=1,
            dropout_rate=0,
            dropout_output=args.dropout_rnn_output,
            concat_layers=False,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding)

        # RNN answer encoder
        choice_input_size = 3 * self.embedding_dim
        self.choice_rnn = layers.StackedBRNN(
            input_size=choice_input_size,
            hidden_size=args.hidden_size,
            num_layers=1,
            dropout_rate=0,
            dropout_output=args.dropout_rnn_output,
            concat_layers=False,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding)

        # Output sizes of rnn encoders
        doc_hidden_size = 2 * args.hidden_size
        question_hidden_size = 2 * args.hidden_size
        choice_hidden_size = 2 * args.hidden_size

        # Answer merging
        self.c_self_attn = layers.LinearSeqAttn(choice_hidden_size)
        self.q_self_attn = layers.LinearSeqAttn(question_hidden_size)

        self.p_q_attn = layers.BilinearSeqAttn(x_size=doc_hidden_size, y_size=question_hidden_size)

        self.p_c_bilinear = nn.Linear(doc_hidden_size, choice_hidden_size)
        self.q_c_bilinear = nn.Linear(question_hidden_size, choice_hidden_size)

        #NLI layer
        self.decomp_atten = layers.DecompAtten(2 * args.hidden_size, 3, 0.01)

    def forward(self, p1, p1_pos, p1_ner, p1_mask, q1, q1_pos, q1_mask, c1, c1_mask, f1_tensor, p1_q1_relation, p1_c1_relation, p2, p2_pos, p2_ner, p2_mask, q2, q2_pos, q2_mask, c2, c2_mask, f2_tensor, p2_q2_relation, p2_c2_relation):
        p1_emb, q1_emb, c1_emb = self.embedding(p1), self.embedding(q1), self.embedding(c1)
        p1_pos_emb, p1_ner_emb, q1_pos_emb = self.pos_embedding(p1_pos), self.ner_embedding(p1_ner), self.pos_embedding(q1_pos)
        p1_q1_rel_emb, p1_c1_rel_emb = self.rel_embedding(p1_q1_relation), self.rel_embedding(p1_c1_relation)

        p2_emb, q2_emb, c2_emb = self.embedding(p2), self.embedding(q2), self.embedding(c2)
        p2_pos_emb, p2_ner_emb, q2_pos_emb = self.pos_embedding(p2_pos), self.ner_embedding(p2_ner), self.pos_embedding(q2_pos)
        p2_q2_rel_emb, p2_c2_rel_emb = self.rel_embedding(p2_q2_relation), self.rel_embedding(p2_c2_relation)

        # Dropout on embeddings
        if self.args.dropout_emb > 0:
            p1_emb = nn.functional.dropout(p1_emb, p=self.args.dropout_emb, training=self.training)
            q1_emb = nn.functional.dropout(q1_emb, p=self.args.dropout_emb, training=self.training)
            c1_emb = nn.functional.dropout(c1_emb, p=self.args.dropout_emb, training=self.training)
            p1_pos_emb = nn.functional.dropout(p1_pos_emb, p=self.args.dropout_emb, training=self.training)
            p1_ner_emb = nn.functional.dropout(p1_ner_emb, p=self.args.dropout_emb, training=self.training)
            q1_pos_emb = nn.functional.dropout(q1_pos_emb, p=self.args.dropout_emb, training=self.training)
            p1_q1_rel_emb = nn.functional.dropout(p1_q1_rel_emb, p=self.args.dropout_emb, training=self.training)
            p1_c1_rel_emb = nn.functional.dropout(p1_c1_rel_emb, p=self.args.dropout_emb, training=self.training)

            p2_emb = nn.functional.dropout(p2_emb, p=self.args.dropout_emb, training=self.training)
            q2_emb = nn.functional.dropout(q2_emb, p=self.args.dropout_emb, training=self.training)
            c2_emb = nn.functional.dropout(c2_emb, p=self.args.dropout_emb, training=self.training)
            p2_pos_emb = nn.functional.dropout(p2_pos_emb, p=self.args.dropout_emb, training=self.training)
            p2_ner_emb = nn.functional.dropout(p2_ner_emb, p=self.args.dropout_emb, training=self.training)
            q2_pos_emb = nn.functional.dropout(q2_pos_emb, p=self.args.dropout_emb, training=self.training)
            p2_q2_rel_emb = nn.functional.dropout(p2_q2_rel_emb, p=self.args.dropout_emb, training=self.training)
            p2_c2_rel_emb = nn.functional.dropout(p2_c2_rel_emb, p=self.args.dropout_emb, training=self.training)

        p1_q1_weighted_emb = self.p_q_emb_match(p1_emb, q1_emb, q1_mask)
        c1_q1_weighted_emb = self.c_q_emb_match(c1_emb, q1_emb, q1_mask)
        c1_p1_weighted_emb = self.c_p_emb_match(c1_emb, p1_emb, p1_mask)
        p1_q1_weighted_emb = nn.functional.dropout(p1_q1_weighted_emb, p=self.args.dropout_emb, training=self.training)
        c1_q1_weighted_emb = nn.functional.dropout(c1_q1_weighted_emb, p=self.args.dropout_emb, training=self.training)
        c1_p1_weighted_emb = nn.functional.dropout(c1_p1_weighted_emb, p=self.args.dropout_emb, training=self.training)
        # print('p_q_weighted_emb', p_q_weighted_emb.size())

        p2_q2_weighted_emb = self.p_q_emb_match(p2_emb, q2_emb, q2_mask)
        c2_q2_weighted_emb = self.c_q_emb_match(c2_emb, q2_emb, q2_mask)
        c2_p2_weighted_emb = self.c_p_emb_match(c2_emb, p2_emb, p2_mask)
        p2_q2_weighted_emb = nn.functional.dropout(p2_q2_weighted_emb, p=self.args.dropout_emb, training=self.training)
        c2_q2_weighted_emb = nn.functional.dropout(c2_q2_weighted_emb, p=self.args.dropout_emb, training=self.training)
        c2_p2_weighted_emb = nn.functional.dropout(c2_p2_weighted_emb, p=self.args.dropout_emb, training=self.training)


        p1_rnn_input = torch.cat([p1_emb, p1_q1_weighted_emb, p1_pos_emb, p1_ner_emb, f1_tensor, p1_q1_rel_emb, p1_c1_rel_emb], dim=2)
        c1_rnn_input = torch.cat([c1_emb, c1_q1_weighted_emb, c1_p1_weighted_emb], dim=2)
        q1_rnn_input = torch.cat([q1_emb, q1_pos_emb], dim=2)
        # print('p_rnn_input', p_rnn_input.size())

        p2_rnn_input = torch.cat([p2_emb, p2_q2_weighted_emb, p2_pos_emb, p2_ner_emb, f2_tensor, p2_q2_rel_emb, p2_c2_rel_emb], dim=2)
        c2_rnn_input = torch.cat([c2_emb, c2_q2_weighted_emb, c2_p2_weighted_emb], dim=2)
        q2_rnn_input = torch.cat([q2_emb, q2_pos_emb], dim=2)

        p1_hiddens = self.doc_rnn(p1_rnn_input, p1_mask)
        c1_hiddens = self.choice_rnn(c1_rnn_input, c1_mask)
        q1_hiddens = self.question_rnn(q1_rnn_input, q1_mask)
        # print('p_hiddens', p_hiddens.size())

        p2_hiddens = self.doc_rnn(p2_rnn_input, p2_mask)
        c2_hiddens = self.choice_rnn(c2_rnn_input, c2_mask)
        q2_hiddens = self.question_rnn(q2_rnn_input, q2_mask)

        ent_proba = self.decomp_atten(c1_hiddens, c2_hiddens)

        q1_merge_weights = self.q_self_attn(q1_hiddens, q1_mask)
        q1_hidden = layers.weighted_avg(q1_hiddens, q1_merge_weights)

        q2_merge_weights = self.q_self_attn(q2_hiddens, q2_mask)
        q2_hidden = layers.weighted_avg(q2_hiddens, q2_merge_weights)

        p1_merge_weights = self.p_q_attn(p1_hiddens, q1_hidden, p1_mask)
        # [batch_size, 2*hidden_size]
        p1_hidden = layers.weighted_avg(p1_hiddens, p1_merge_weights)
        # print('p_hidden', p_hidden.size())

        p2_merge_weights = self.p_q_attn(p2_hiddens, q2_hidden, p2_mask)
        # [batch_size, 2*hidden_size]
        p2_hidden = layers.weighted_avg(p2_hiddens, p2_merge_weights)
        # print('p_hidden', p_hidden.size())

        c1_merge_weights = self.c_self_attn(c1_hiddens, c1_mask)
        # [batch_size, 2*hidden_size]
        c1_hidden = layers.weighted_avg(c1_hiddens, c1_merge_weights)
        # print('c_hidden', c_hidden.size())

        c2_merge_weights = self.c_self_attn(c2_hiddens, c2_mask)
        # [batch_size, 2*hidden_size]
        c2_hidden = layers.weighted_avg(c2_hiddens, c2_merge_weights)
        # print('c_hidden', c_hidden.size())

        logits1 = torch.sum(self.p_c_bilinear(p1_hidden) * c1_hidden, dim=-1)
        logits1 += torch.sum(self.q_c_bilinear(q1_hidden) * c1_hidden, dim=-1)
        proba1 = F.sigmoid(logits1)

        logits2 = torch.sum(self.p_c_bilinear(p2_hidden) * c2_hidden, dim=-1)
        logits2 += torch.sum(self.q_c_bilinear(q2_hidden) * c2_hidden, dim=-1)
        proba2 = F.sigmoid(logits2)
        # print('proba', proba.size())

        return proba1, proba2, ent_proba

