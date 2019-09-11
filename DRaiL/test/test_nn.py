"""
Cosmo Zhang @ Purdue July/2016
Filename: test_nn.py
For the Probalistic Soft Logic
coding: -*- utf-8 -*-
"""

import json
import sys
from pprint import pprint
from drail.neuro.nn_model import NeuralNetworks
import numpy as np

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python neuro_build.py [configuaration file]")
    conf_f_name = sys.argv[1]

    with open(conf_f_name, "r") as f:
        config = json.load(f)
    nn = NeuralNetworks(config["models"][0], ".", 0)
    nn.setdim([100], 10)
    # added this calls to build and compile when used (removed from constructor)
    nn.build_architecture()
    nn.compile_functions()
    # pprint(config)

    # '''
    # Pseudo data for debugging
    Train_X = np.random.randn(200, 100).astype("float32")
    Train_Y = np.random.randint(0, 10, size = (200)).astype("int32")

    Dev_X = np.random.randn(20, 100).astype("float32")
    Dev_Y = np.random.randint(0, 10, size = (20)).astype("int32")

    Test_X = np.random.randn(20, 100).astype("float32")
    Test_Y = np.random.randint(0, 10, size = (20)).astype("int32")
    # '''

    nn.train_on_epoches(Train_X, Train_Y, Dev_X, Dev_Y)



if __name__ == '__main__':
    main()
