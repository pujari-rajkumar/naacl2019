import torch
import numpy as np

class NeuralHub(torch.nn.Module):

    def __init__(self, nn_list, learning_rate, use_gpu,
                 l1_lambda, l2_lambda):
        super(NeuralHub, self).__init__()
        self.potentials = torch.nn.ModuleList(nn_list)
        self.lr = learning_rate
        self.use_gpu = use_gpu
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def _l1_penalty(self):
        if self.use_gpu:
            l1_norm = torch.autograd.Variable(torch.cuda.FloatTensor([0])).cuda()
        else:
            l1_norm = torch.autograd.Variable(torch.FloatTensor([0]))

        for W in self.parameters():
            l1_norm += torch.abs(W).sum()
        return l1_norm

    def _l2_penalty(self):
        if self.use_gpu:
            l2_norm = torch.autograd.Variable(torch.cuda.FloatTensor([0])).cuda()
        else:
            l2_norm = torch.autograd.Variable(torch.FloatTensor([0]))
        for W in self.parameters():
            l2_norm += torch.sqrt(torch.pow(W, 2).sum())
        return l2_norm

    def _potential_scores(self, index, X):
        output = self.potentials[index](X)
        return output

    def _score_instance(self, X_ls, y_ls, y_index_ls):
        if self.use_gpu:
            score = torch.autograd.Variable(torch.cuda.FloatTensor([0])).cuda()
        else:
            score = torch.autograd.Variable(torch.FloatTensor([0]))

        for i in range(0, len(self.potentials)):
                if len(X_ls[i]) == 0:
                    continue
                X = X_ls[i]; y = y_ls[i]; y_index = y_index_ls[i]
                potential_scores = self._potential_scores(i, X)
                score += potential_scores[y_index, y].sum()
        return score

    def loss(self, X_ls, y_pred_ls, y_pred_index_ls,
             y_gold_ls, y_gold_index_ls):
        pred_score = self._score_instance(X_ls, y_pred_ls, y_pred_index_ls)
        gold_score = self._score_instance(X_ls, y_gold_ls, y_gold_index_ls)

        return pred_score - gold_score +\
               self.l1_lambda * self._l1_penalty() +\
               self.l2_lambda * self._l2_penalty()
