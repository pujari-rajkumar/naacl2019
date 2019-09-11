import torch
import numpy as np
from nn_model import NeuralNetworks, ModelType
import torch.nn.functional as F

class Shared1HLx2inputs(NeuralNetworks):

    def __init__(self, config, nn_id, use_gpu, output_dim):
        super(Shared1HLx2inputs, self).__init__(config, nn_id)
        self.type = ModelType.FF
        self.use_gpu = use_gpu
        self.output_dim = output_dim

    def build_architecture(self, rule_template, fe, shared_layers):
        self.minibatch_size = self.config["batch_size"]
        n_hidden = shared_layers['hidden_layer_1']['nout']

        self.layer1 = shared_layers['hidden_layer_1']['layer']
        self.layer2 = torch.nn.Linear(n_hidden*2, self.output_dim)

    def forward(self, x):
        x1 = [elem[0] for elem in x['input']]
        x2 = [elem[1] for elem in x['input']]
        tensor1 = self._get_float_tensor(x1)
        tensor2 = self._get_float_tensor(x2)

        var1 = self._get_grad_variable(tensor1)
        var2 = self._get_grad_variable(tensor2)

        hidden1 = self.layer1(var1)
        hidden2 = self.layer1(var2)

        out = F.relu(torch.cat([hidden1, hidden2], 1))
        out = self.layer2(out)
        return F.softmax(out)
