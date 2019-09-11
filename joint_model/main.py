import os
import time
import torch
import random
import numpy as np

from datetime import datetime

from utils import load_data, build_vocab
from config import args
from model import Model

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if __name__ == '__main__':

    build_vocab()
    #train_data = load_data('./data/train-data-processed.json', './data/train-data-nli.txt')
    #dev_data = load_data('./data/dev-data-processed.json', './data/dev-data-nli.txt')
    train_data = load_data('./data/train_456-fixedIds-processed.json', './data/train_456-data-nli.txt')
    dev_data = load_data('./data/train_456-fixedIds-processed.json', './data/dev_456-data-nli.txt')
    if args.test_mode:
        # use validation data as training data
        train_data += dev_data
        dev_data = []
    args.pretrained = './checkpoint/1234-2018-12-09T22:21:32.064766.mdl'
    model = Model(args)

    best_dev_acc = 0.0
    os.makedirs('./checkpoint', exist_ok=True)
    checkpoint_path = './checkpoint/%d-%s.mdl' % (args.seed, datetime.now().isoformat())
    print('Trained model will be saved to %s' % checkpoint_path)

    for i in range(args.epoch):
        print('Epoch %d...' % i)
        if i == 0:
            dev_acc = model.evaluate(dev_data, debug=False, eval_multirc=True)
            print('Dev accuracy: %f' % dev_acc)
        start_time = time.time()
        choice_data, nli_data = train_data
        np.random.shuffle(nli_data)
        cur_train_data = (choice_data, nli_data)

        model.train(cur_train_data)
        dev_acc = model.evaluate(dev_data, debug=False, eval_multirc=True)
        print('Dev accuracy: %f' % dev_acc)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            print('Saving model..')
            model.save(checkpoint_path)
        elif args.test_mode:
            model.save(checkpoint_path)
        print('Epoch %d use %d seconds.' % (i, time.time() - start_time))

    print('Best dev accuracy: %f' % best_dev_acc)
