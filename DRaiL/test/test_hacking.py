"""
Does not work...
"""
import json
import sys
from pprint import pprint
import numpy as np
import lasagne
from lasagne import layers
import theano
import theano.tensor as T
from lasagne.utils import floatX


class NeuralNetworks(object):

    def __init__(self):

        self.input_dim = [10]
        self.input_var = T.fmatrix("input")
        self.target_var = T.ivector('targets')
        self.b = theano.shared(value = floatX([1e+16, -1e+16, -1e+16]))

    def build_architecture(self):

        l_out = layers.InputLayer(shape = tuple([None] + self.input_dim), input_var = self.input_var)
        l_out = layers.DenseLayer(l_out, num_units = 10, nonlinearity = lasagne.nonlinearities.sigmoid)

        self.l_out1 = l_out

        # 3 output classes
        # l_out = layers.DenseLayer(l_out, num_units = 3, nonlinearity = lasagne.nonlinearities.softmax)
        print self.b.get_value()
        l_out = layers.DenseLayer(incoming = l_out, num_units = 3, b = self.b, nonlinearity = lasagne.nonlinearities.softmax)

        # print type(self.b)
        self.l_out = l_out

    def compile_functions(self):
        prediction = layers.get_output(self.l_out)
        intermediate = layers.get_output(self.l_out1)
        loss = lasagne.objectives.categorical_crossentropy(prediction, self.target_var).mean()
        params = layers.get_all_params(self.l_out, trainable=True)

        updates = lasagne.updates.sgd(loss, params, learning_rate = 0.001)

        self.update_fn = theano.function([self.input_var, self.target_var], [loss], updates=updates)

        self.pred_fn = theano.function([self.input_var], [prediction, intermediate])

    def train(self):
        pass


def main():

    nn = NeuralNetworks()
    # added this calls to build and compile when used (removed from constructor)
    nn.build_architecture()
    nn.compile_functions()
    # pprint(config)

    # '''
    # Pseudo data for debugging
    Train_X = np.random.randn(5, 10).astype("float32")
    Train_Y = np.random.randint(0, 3, size = (5)).astype("int32")

    Dev_X = np.random.randn(2, 10).astype("float32")
    Dev_Y = np.random.randint(0, 3, size = (2)).astype("int32")

    Test_X = np.random.randn(2, 10).astype("float32")
    Test_Y = np.random.randint(0, 3, size = (2)).astype("int32")
    # '''

    results, im_results = nn.pred_fn(Train_X)
    print results, im_results

    print Train_Y


if __name__ == '__main__':
    main()

