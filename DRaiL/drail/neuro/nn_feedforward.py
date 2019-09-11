import torch
import numpy as np
from nn_model import NeuralNetworks, ModelType

class FeedForwardNetwork(NeuralNetworks):

    def __init__(self, config, nn_id, use_gpu, output_dim):
        super(FeedForwardNetwork, self).__init__(config, nn_id)
        self.layers = torch.nn.ModuleList([])
        self.type = ModelType.FF
        self.use_gpu = use_gpu
        self.output_dim = output_dim

    def build_architecture(self, rule_template, fe, shared_layers={}):
        l_out_dim = rule_template.feat_vector_sz
        layers_conf = self.config["layers"]
        self.minibatch_size = self.config["batch_size"]

        for layer_conf in layers_conf[:-1]:
            if 'n_hidden' in layer_conf:
                n_hidden = layer_conf['n_hidden']

            if layer_conf["ltype"] == "Dense":
                if self.use_gpu:
                    self.layers.append(torch.nn.Linear(l_out_dim, n_hidden).cuda())
                else:
                    self.layers.append(torch.nn.Linear(l_out_dim, n_hidden))
                #print l_out_dim, n_hidden
                if layer_conf['act_func'] == "relu":
                    self.layers.append(torch.nn.ReLU())
                elif layer_conf['act_func'] == "tanh":
                    self.layers.append(torch.nn.Tanh())
                elif layer_conf['act_func'] == "sigmoid":
                    self.layers.append(torch.nn.Sigmoid())
                elif layer_conf['act_func'] == "softmax":
                    self.layers.append(torch.nn.Softmax())
                l_out_dim = n_hidden

        out_layer_conf = layers_conf[-1]
        if self.use_gpu:
            self.layers.append(torch.nn.Linear(l_out_dim, self.output_dim).cuda())
        else:
            self.layers.append(torch.nn.Linear(l_out_dim, self.output_dim))
        #print l_out_dim, self.output_dim
        if out_layer_conf['act_func'] == "relu":
            self.layers.append(torch.nn.ReLU())
        elif out_layer_conf['act_func'] == "tanh":
            self.layers.append(torch.nn.Tanh())
        elif out_layer_conf['act_func'] == "sigmoid":
            self.layers.append(torch.nn.Sigmoid())
        elif out_layer_conf['act_func'] == "softmax":
            self.layers.append(torch.nn.Softmax())
        #print

    def forward(self, x):
        tensor = self._get_float_tensor(x['vector'])
        var = self._get_grad_variable(tensor)
        for i, l in enumerate(self.layers):
            var = l(var)
        return var

