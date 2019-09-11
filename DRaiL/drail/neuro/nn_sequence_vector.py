import torch
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

from nn_model import NeuralNetworks, ModelType

class SequenceVectorNet(NeuralNetworks):

    def __init__(self, config, nn_id, use_gpu, output_dim):
        super(SequenceVectorNet, self).__init__(config, nn_id)
        #self.type = ModelType.Sequence
        self.use_gpu = use_gpu
        self.output_dim = output_dim

    def build_architecture(self, rule_template, fe, shared_layers={}):
        self.minibatch_size = self.config['batch_size']
        #embedding_layers, emb_dims = \
        #        self._embedding_inputs(rule_template, fe, use_vocab_index=False)
        # embedding layer for sequence
        #self.sequence_emb = embedding_layers[self.config['input_sequence']]

        # embedding layer for vector
        #self.vector_emb = embedding_layers[self.config['input_vector']]

        # hidden layer and input information
        self.n_hidden_sequence = self.config["n_hidden_sequence"]
        self.n_hidden_vector = self.config["n_hidden_vector"]
        self.n_input_sequence = self.config['n_input_sequence']
        self.n_input_vector = self.config['n_input_vector']

        # LSTM for the sequence
        self.sequence_lstm =\
                torch.nn.LSTM(input_size=self.n_input_sequence,
                              hidden_size=self.n_hidden_sequence,
                              bidirectional=True,
                              batch_first=True)

        # input for the vector
        self.vector_input =\
                torch.nn.Linear(self.n_input_vector, self.n_hidden_vector)

        '''
        self.concat2hidden =\
                torch.nn.Linear(self.n_hidden_sequence*2 + self.n_hidden_vector,
                           self.n_hidden_vector)
        '''

        self.dropout_layer = torch.nn.Dropout(p=0.5)

        # hidden layer for both representations
        self.hidden2label =\
                torch.nn.Linear(self.n_hidden_sequence*2 + self.n_hidden_vector,
                                self.output_dim)

        if self.use_gpu:
            self.sequence_lstm = self.sequence_lstm.cuda()
            self.vector_input = self.vector_input.cuda()
            #self.concat2hidden = self.concat2hidden.cuda()
            self.dropout_layer = self.dropout_layer.cuda()
            self.hidden2label = self.hidden2label.cuda()

        self.hidden_bilstm = self.init_hidden_bilstm()

    def init_hidden_bilstm(self):
        var1 = torch.autograd.Variable(torch.zeros(2, self.minibatch_size,
                                                    self.n_hidden_sequence))
        var2 = torch.autograd.Variable(torch.zeros(2, self.minibatch_size,
                                                    self.n_hidden_sequence))

        if self.use_gpu:
            var1 = var1.cuda()
            var2 = var2.cuda()

        return (var1, var2)

    def forward(self, x):
        seqs = [elem[0] for elem in x['input']]

        # get the length of each seq in your batch
        seq_lengths = self._get_long_tensor(map(len, seqs))
        # dump padding everywhere, and place seqs on the left.
        # NOTE: you only need a tensor as big as your longest sequence
        max_seq_len = seq_lengths.max()

        tensor_seq = torch.zeros((len(seqs), max_seq_len, self.n_input_sequence)).float()
        if self.use_gpu:
            tensor_seq = tensor_seq.cuda()
        for idx, (seq, seqlen) in enumerate(zip(seqs, seq_lengths)):
            tensor_seq[idx, :seqlen] = self._get_float_tensor(seq)

        var_seq = self._get_variable(tensor_seq)
        seq_lengths = self._get_variable(seq_lengths)

        # pack padded sequences
        packed_input_seq = pack_padded_sequence(var_seq, list(seq_lengths.data), batch_first=True)
        # run lstm over sequence
        self.hidden_bilstm = self.init_hidden_bilstm()
        packed_output, self.hidden_bilstm = \
                self.sequence_lstm(packed_input_seq, self.hidden_bilstm)
        # unpack the output
        unpacked_output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # extract last timestep, since doing [-1] would get the padded zeros
        idx = (seq_lengths - 1).view(-1, 1).expand(
            unpacked_output.size(0), unpacked_output.size(2)).unsqueeze(1)
        lstm_output = unpacked_output.gather(1, idx).squeeze()

        if len(list(lstm_output.size())) == 1:
            lstm_output = lstm_output.unsqueeze(0)

        # get the vectors
        vectors = [elem[1] for elem in x['input']]
        tensor_vec = self._get_float_tensor(vectors)
        var_vec = self._get_variable(tensor_vec)

        # run vector over hidden layer
        hidden_vector = self.vector_input(var_vec)
        hidden_vector = F.relu(hidden_vector)

        # join sequence and vector
        lstm_output = self.dropout_layer(lstm_output)
        concat = torch.cat([lstm_output, hidden_vector], 1)

        #print out.size()

        out = self.hidden2label(concat)
        out = F.softmax(out)
        return out

