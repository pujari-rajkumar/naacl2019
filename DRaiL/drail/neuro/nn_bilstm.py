import torch
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from nn_model import NeuralNetworks, ModelType


class BiLSTM(NeuralNetworks):

    def __init__(self, config, nn_id, use_gpu, output_dim):
        super(BiLSTM, self).__init__(config, nn_id)
        self.type = ModelType.Sequence
        self.use_gpu = use_gpu
        self.output_dim = output_dim

    def build_architecture(self, rule_template, fe, shared_layers={}):
        self.embeddings, self.emb_dims = self._embedding_inputs(rule_template, fe)
        self.input_lengths = sum(self.emb_dims.values()) + rule_template.feat_vector_sz

        self.num_layers = self.config['n_layers']
        self.minibatch_size = self.config['batch_size']
        self.hidden_dim = self.config['hidden_dim']

        self.lstm = torch.nn.LSTM(input_size=self.input_lengths,
                                  hidden_size=self.hidden_dim,
                                  bidirectional=True)

        self.hidden2label = torch.nn.Linear(self.hidden_dim*2, self.output_dim)

        if self.use_gpu:
            self.lstm = self.lstm.cuda()
            self.hidden2label = self.hidden2label.cuda()

        self.hidden = self.init_hidden()

    def init_hidden(self):
        var1 = torch.autograd.Variable(torch.zeros(self.num_layers*2,
                                                    self.minibatch_size,
                                                    self.hidden_dim))
        var2 = torch.autograd.Variable(torch.zeros(self.num_layers*2,
                                                    self.minibatch_size,
                                                    self.hidden_dim))

        if self.use_gpu:
            var1 = var1.cuda()
            var2 = var2.cuda()

        return (var1, var2)

    def forward(self, sequence, i):
        # assuming batch 1 for now, will need to deal with padding
        self.hidden = self.init_hidden()
        all_embeds = []

        combined = None
        for emb in sequence['embedding']:
            tensor = self._get_long_tensor(sequence['embedding'][emb][i])
            var = self._get_variable(tensor)
            embeds = self.embeddings[emb](var)
            print embeds.data.shape
            all_embeds.append(embeds)
            #sequence = sequence.transpose(0,1) # (B,L,D) -> (L,B,D)
            #packed_input = pack_padded_sequence(embeds, seq_len.cpu().numpy())
            #packed_out, self.hidden = self.lstm(packed_input, self.hidden)
            #lstm_out, _ = pad_packed_sequence(packed_out)
        if len(all_embeds) > 0:
            combined = torch.cat(all_embeds, 1)

        if len(sequence['vector']) > 0:
            tensor = self._get_float_tensor(sequence['vector'][i])
            tensor = tensor.transpose(0,1)
            var = self._get_grad_variable(tensor)
            if combined is not None:
                combined = torch.cat((combined, var), 1)
            else:
                combined = var

        combined = combined.view(combined.data.shape[0], self.minibatch_size, -1)

        lstm_out, self.hidden = self.lstm(combined, self.hidden)
        y = self.hidden2label(lstm_out[-1])
        log_probs = torch.nn.functional.softmax(y)

        return log_probs
