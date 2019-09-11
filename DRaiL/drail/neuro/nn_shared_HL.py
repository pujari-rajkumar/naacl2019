import torch
import numpy as np
from nn_model import NeuralNetworks, ModelType

class Shared1HL(NeuralNetworks):

    def __init__(self, config, nn_id, use_gpu, output_dim):
        super(Shared1HL, self).__init__(config, nn_id)
        self.type = ModelType.FF
        self.use_gpu = use_gpu
        self.output_dim = output_dim

    def build_architecture(self, rule_template, fe, shared_layers):
        self.minibatch_size = self.config["batch_size"]
        n_hidden = shared_layers['hidden_layer_1']['nout']

        layer1 = shared_layers['hidden_layer_1']['layer']
        layer2 = torch.nn.Linear(n_hidden, self.output_dim)

        self.layers = torch.nn.ModuleList([
                layer1,
                torch.nn.ReLU(),
                layer2,
                torch.nn.Softmax()])

    def forward(self, x):
        tensor = self._get_float_tensor(x['vector'])
        var = self._get_grad_variable(tensor)
        for i, l in enumerate(self.layers):
            var = l(var)
        return var

