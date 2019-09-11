import torch
import numpy as np
from ..model.feature_function import FeatureType

class NeuralNetworks(torch.nn.Module):

    def __init__(self, config, nn_id):
        super(NeuralNetworks, self).__init__()
        self.config = config
        self.nn_id = nn_id

    def build_architecture(self, rule_template, fe, shared_layers={}):
        raise NotImplementedError

    def _get_float_tensor(self, inp):
        if self.use_gpu:
            return torch.cuda.FloatTensor(inp)
        return torch.FloatTensor(inp)

    def _get_long_tensor(self, inp):
        if self.use_gpu:
            return torch.cuda.LongTensor(inp)
        return torch.LongTensor(inp)

    def _get_grad_variable(self, tensor):
        var = torch.autograd.Variable(tensor, requires_grad=True)
        if self.use_gpu:
            var = var.cuda()
        return var

    def _get_variable(self, tensor):
        var = torch.autograd.Variable(tensor)
        if self.use_gpu:
            var = var.cuda()
        return var

    def _l1_penalty(self):
        if self.use_gpu:
            l1_norm = torch.autograd.Variable(torch.cuda.FloatTensor([0]))
            l1_norm = l1_norm.cuda()
        else:
            l1_norm = torch.autograd.Variable(torch.FloatTensor([0]))

        for W in self.parameters():
            l1_norm += torch.abs(W).sum()
        return l1_norm

    def _l2_penalty(self):
        if self.use_gpu:
            l2_norm = torch.autograd.Variable(torch.cuda.FloatTensor([0]))
            l2_norm = l2_norm.cuda()
        else:
            l2_norm = torch.autograd.Variable(torch.FloatTensor([0]))

        for W in self.parameters():
            l2_norm += torch.sqrt(torch.pow(W, 2).sum())
        return l2_norm

    def _pretrained_embedding(self, vocab_size, emb_dim, emb_container, vocab_index=None):
        pretrained_embeddings = np.random.uniform(-0.0025, 0.0025, (vocab_size, emb_dim))
        for word, vector in emb_container.items():
            if vocab_index is not None and word in vocab_index:
                pretrained_embeddings[vocab_index[word]] = vector
            else:
                pretrained_embeddings[word] = vector
        return pretrained_embeddings

    def _embedding_inputs(self, rule_template, fe, use_vocab_index=True):
        emb_inputs = {}
        emb_dims = {}
        for feat in rule_template.feat_functions:
            if feat.feat_type == FeatureType.Embedding:

                # istantiate all elements from feature class
                vocab_size = fe.__dict__[feat.vocab_size]
                emb_dim = fe.__dict__[feat.embedding_size]
                emb_container = fe.__dict__[feat.embedding_container]
                vocab_index = fe.__dict__[feat.vocab_index]

                # create embedding layer and keep track of dimensions
                inp = torch.nn.Embedding(vocab_size, emb_dim)
                emb_dims[feat.feat_name] = emb_dim

                # load pretrained embeddings
                if emb_container is not None:
                    if use_vocab_index:
                        pretrained_embeddings = \
                                self._pretrained_embedding(vocab_size, emb_dim,
                                                           emb_container, vocab_index)
                    else:
                        pretrained_embeddings = \
                                self._pretrained_embedding(vocab_size, emb_dim,
                                                           emb_container)

                    inp.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

                if self.use_gpu:
                    inp = inp.cuda()

                emb_inputs[feat.feat_name] = inp

        return emb_inputs, emb_dims

    def global_loss(self, inp, y_pred, y_gold):
        output = self(inp)

        pred_score = torch.autograd.Variable(torch.FloatTensor([0]))
        gold_score = torch.autograd.Variable(torch.FloatTensor([0]))

        if self.use_gpu:
            pred_score = pred_score.cuda()
            gold_score = gold_score.cuda()
        n = np.asarray(range(0, self.minibatch_size), dtype=np.int64)

        pred_score += (output[n, y_pred]).sum()
        gold_score += (output[n, y_gold]).sum()

        l1_lambda = 0.0; l2_lambda = 0.0
        if 'l1_lambda' in self.config:
            l1_lambda = self.config['l1_lambda']
        if 'l2_lambda' in self.config:
            l2_lambda = self.config['l2_lambda']

        return (pred_score - gold_score) + \
                l1_lambda * self._l1_penalty() + \
                l2_lambda * self._l2_penalty()

    def pytorch_loss(self, y_pred, Y, loss_fn):
        if self.use_gpu:
            y = torch.cuda.LongTensor(Y)
            y_gold = torch.autograd.Variable(y).cuda()
        else:
            y = torch.LongTensor(Y)
            y_gold = torch.autograd.Variable(y)
        return loss_fn(y_pred, y_gold)


class ModelType(object):
    FF, Sequence = range(0, 2)
