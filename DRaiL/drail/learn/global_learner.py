import numpy as np
import os
import json
import sys
from collections import OrderedDict
import torch
import time
import random
import time
from sklearn.metrics import *
from learner import Learner
from metrics import Metrics
from ..model.rule import RuleGrounding
from ..neuro.nn_model_global import NeuralHub
from ..neuro.nn_feedforward import FeedForwardNetwork
from ..neuro.nn_bilstm import BiLSTM
from ..neuro import nn_utils
from ..inference.ilp_inferencer import *


class GlobalLearner(Learner):

    def __init__(self, learning_rate):
        super(GlobalLearner, self).__init__()
        self.learning_rate = learning_rate
        self.reset()
        self.timestamp = time.time()

    def reset(self):
        super(GlobalLearner, self).reset()
        self.brain = None; self.optimizer = None
        self.train_metrics = Metrics()
        self.dev_metrics = Metrics()
        self.test_metrics = Metrics()

    def build_models(self, db, config_path, isdic=False):
        brain_nns_ls = []
        self._build_models(db, config_path, brain_nns_ls, isdic)
        # TO-DO pass lambda parameters as input to GL
        self.brain = NeuralHub(brain_nns_ls, learning_rate=self.learning_rate, use_gpu=False,
                               l1_lambda=0, l2_lambda=0)
        self.optimizer = torch.optim.SGD(self.brain.parameters(), lr=self.brain.lr)

    def extract_data(self, db, train_filters=[], dev_filters=[], test_filters=[]):
        # extract data once
        print "Extracting train rules..."
        self.instances = {}; self.gold_heads = {}
        self.instance_groundings = {}; self.constraint_groundings = {}
        self.constraint_heads = {}

        self.instances['train'], self.instance_groundings['train'],\
        self.gold_heads['train'] = \
                self.extract_rules(db, train_filters, is_global_train=True)
        self.constraint_groundings['train'], self.constraint_heads['train'] = \
                self.extract_constraints(db, train_filters, is_global_train=True)
        print "Extracting dev rules..."
        self.instances['dev'], self.instance_groundings['dev'],\
        self.gold_heads['dev'] = \
                self.extract_rules(db, dev_filters)
        self.constraint_groundings['dev'], _ = \
                self.extract_constraints(db, dev_filters)
        print "Extracting test rules..."
        self.instances['test'], self.instance_groundings['test'],\
        self.gold_heads['test'] = \
                self.extract_rules(db, test_filters)
        self.constraint_groundings['test'], _ = \
                self.extract_constraints(db, test_filters)

    def extract_observed_data(self, db, train_filters=[], dev_filters=[], test_filters=[]):
        # extract training data
        observed_train = self.get_observed_data(
                db, train_filters)
        observed_dev = self.get_observed_data(
                db, dev_filters)
        observed_test = self.get_observed_data(
                db,  test_filters)
        return observed_train, observed_dev, observed_test

    def get_structures(self, instance_groundings, active_heads):
        y_ls = []; y_index_ls = []
        for ruleidx, rule_template in enumerate(self.ruleset['rule']):
            for index in range(0, len(self.nnindex_ls[ruleidx])):
                y = []; y_index = []
                bgs = instance_groundings[ruleidx][index]
                for b_index, body in enumerate(bgs):
                    # if the rule is active
                    if len(bgs[body]['unobs'] & active_heads) == len(bgs[body]['unobs']):
                        # binary head
                        if not rule_template.head.isobs:
                            y.append(int(bgs[body]['heads'][0] in active_heads))
                        else:
                            for head, head_index in zip(bgs[body]['heads'], bgs[body]['heads_index']):
                                if head in active_heads:
                                    y.append(head_index)
                        y_index.append(b_index)
                y_ls.append(np.asarray(y).astype(np.int64))
                y_index_ls.append(np.asarray(y_index).astype(np.int64))
        return y_ls, y_index_ls

    def hot_start(self, db, observed_train, observed_dev):
        for ruleidx, rule_template in enumerate(self.ruleset['rule']):
            for index, nnidx in enumerate(self.nnindex_ls[ruleidx]):
                nnet = self.brain.potentials[nnidx]
                X_train, Y_train = observed_train[ruleidx][index]
                X_dev, Y_dev = observed_dev[ruleidx][index]
                nn_utils.train_local_on_epoches(nnet, X_train, Y_train, X_dev, Y_dev, None, None)

    def train(self, db, train_filters, dev_filters, test_filters, opt_predicate, n_epoch=1,
              hot_start=False, loss_augmented_inference=False, K=None):

        if hot_start:
            # get observed data to check local nets
            # and hot start each instance if hs is on
            observed_train, observed_dev, observed_test = \
            self.extract_observed_data(db, train_filters, dev_filters, test_filters)
            self.hot_start(db, observed_train, observed_dev)

        # get gold structures for the training set
        y_gold_ls = {}; y_gold_index_ls = {}
        for i, inst in enumerate(self.instances['train']):
            y_gold_ls_i, y_gold_index_ls_i = \
                    self.get_structures(self.instance_groundings['train'][inst],
                            set([RuleGrounding.pred_str(h) for h in self.gold_heads['train'][inst]]) | self.constraint_heads['train'][inst])
            y_gold_ls[inst] = y_gold_ls_i
            y_gold_index_ls[inst] = y_gold_index_ls_i

        loss = None
        best_val_measure = -float("infinity")
        done_looping = False
        patience_counter = 0

        epoch = 1
        random.seed(1234)
        while not done_looping:
            #print "training..."

            for i, inst in enumerate(self.instances['train']):
                # run inference

                '''
                if len(self.instance_groundings_tr[inst][0][0].keys()) == 0:
                    continue
                '''

                active_heads, xs_ls, train_metrics = \
                        self.predict(self.instance_groundings['train'][inst],
                                self.gold_heads['train'][inst],
                                self.constraint_groundings['train'][inst],
                                get_active_heads=True,
                                loss_augmented_inference=loss_augmented_inference,
                                K=K)

                if len(active_heads) == 0:
                    continue

                y_pred_ls_i, y_pred_index_ls_i = \
                        self.get_structures(self.instance_groundings['train'][inst], active_heads)



                matches = 0
                for gold, pred in zip(y_gold_ls[inst][0], y_pred_ls_i[0]):
                    matches += int(gold == pred)

                loss = nn_utils.backpropagate_global(
                        self.brain, self.optimizer, xs_ls,
						y_pred_ls_i, y_pred_index_ls_i,
						y_gold_ls[inst], y_gold_index_ls[inst])

                #print i, ": ", inst, " loss: ", loss

            #print "Running inference for dev set", len(self.instances['dev'])
            for i, inst in enumerate(self.instances['dev']):
                #print "Number of constraints", len(self.constraint_groundings['dev'][inst])
                dev_active_heads, _, dev_metrics = \
                        self.predict(self.instance_groundings['dev'][inst],
                                self.gold_heads['dev'][inst],
                                self.constraint_groundings['dev'][inst], get_active_heads = True,
                                K=K)
                self.dev_metrics.load_metrics(dev_metrics)

                #for head in dev_active_heads:
                #    print head


			# this is hardcoded here, ideally we want to separate by predicate
            dev_f1 = f1_score(self.dev_metrics.metrics[opt_predicate]["gold_data"], self.dev_metrics.metrics[opt_predicate]["pred_data"], average='macro')
            print 'epoch', epoch, "dev_f1", dev_f1

            if dev_f1 > best_val_measure:
                patience_counter = 0
                best_val_measure = dev_f1
                best_epoch = epoch
                model_state = self.brain.state_dict()
                torch.save(model_state, 'best_model_{0}.pt'.format(self.timestamp))
            else:
                patience_counter += 1
                # hardcoded for now
                if patience_counter >= 15:
                    done_looping = True

            # get local metrics for all instances (hacking only local classifiers)
            '''
            for i in range(1):
                for ruleidx in range(1):
                    for nnidx in self.nnindex_ls[ruleidx]:
                        nnet = self.brain.neuralnetworks[nnidx]
                        train_acc = nnet.test(observed_train_ls[i][ruleidx][nnidx])
                        dev_acc = nnet.test(observed_dev_ls[i][ruleidx][nnidx])
                        test_acc = nnet.test(observed_test_ls[i][ruleidx][nnidx])
                        print "\tlocal train_acc", train_acc, "test_acc", test_acc, "dev_acc", dev_acc
            '''

            # reset metrics
            self.dev_metrics = Metrics()
            self.test_metrics = Metrics()
            epoch += 1

        print "BEST F1", best_val_measure
        self.brain.load_state_dict(torch.load('best_model_{0}.pt'.format(self.timestamp)))

        # predict the final thing
        for i, inst in enumerate(self.instances['test']):
            # run inference
            test_active_heads, _, test_metrics = \
                    self.predict(self.instance_groundings['test'][inst],
                            self.gold_heads['test'][inst],
                            self.constraint_groundings['test'][inst], get_active_heads = True,
                            K=K)
            self.test_metrics.load_metrics(test_metrics)

        return self.test_metrics

    def predict(self, instance_groundings,
                gold_heads, constraint_groundings,
                get_active_heads=True, time_limit=None,
                loss_augmented_inference=False, K=None):

        inference_weights = OrderedDict()
        xs_ls = []

        if loss_augmented_inference:
            gold_heads_str = [RuleGrounding.pred_str(h) for h in gold_heads]

        for ruleidx, rule_template in enumerate(self.ruleset['rule']):
            #print "Rule", rule_template
            if K is None:
                K = rule_template.head_cardinality()
            else:
                K = min(K, rule_template.head_cardinality())

            for index, nnidx in enumerate(self.nnindex_ls[ruleidx]):
                #print "NNet", index, nnidx

                X = {'vector': [], 'embedding': {}, 'input': []}; y_pred = []; pivots = []

                if len(instance_groundings[ruleidx][index].keys()) > 0:

                    # predict with neural nets
                    nnet = self.brain.potentials[nnidx]
                    for body in instance_groundings[ruleidx][index]:
                        X = self._add_features(
                                instance_groundings[ruleidx][index][body]['feat_repr'],
                                X)
                    y_pred = nn_utils.predict_local_scores(nnet, X)

                    # we want only the top k solutions, find the pivots to discriminate
                    partial_order = np.partition(y_pred, y_pred.shape[1] - K)
                    pivots = partial_order[:,y_pred.shape[1] - K]
                    xs_ls.append(X)
                else:
                    xs_ls.append([])

                # instantiate weights
                for body, scores, pivot in zip(instance_groundings[ruleidx][index], y_pred, pivots):
                #for body, scores in zip(instance_groundings[ruleidx][index], y_pred):
                    for gr_i, gr in enumerate(instance_groundings[ruleidx][index][body]['groundings']):
                        if gr.is_binary_head:
                            idx_in_nn_out = 1
                        else:
                            idx_in_nn_out = instance_groundings[ruleidx][index][body]['heads_index'][gr_i]
                        weight = scores[idx_in_nn_out]

                        # if weight not in the top K solutions, do not add it to the problem
                        if weight >= pivot:
                            inference_weights[gr] = weight * rule_template.lambda_
                            if loss_augmented_inference and RuleGrounding.pred_str(gr.head) in gold_heads_str:
                                inference_weights[gr] += 1
                        #inference_weights[gr] = weight * instance_groundings[ruleidx][index][body]['lambda']


        start = time.time()
        inferencer = ILPInferencer(inference_weights, constraint_groundings)
        inferencer.encode()
        inferencer.optimize()
        end_encode = time.time()
        inferencer.optimize()
        end = time.time()
        #print "encoding", end_encode - start
        #print "inference", end - end_encode
        #exit()

        metrics = inferencer.evaluate(gold_heads, binary_predicates=self.binary_classifiers)

        if get_active_heads:
            active_heads = inferencer.get_predictions()
            #print "matches: ", len(active_heads & gold_heads), "/", len(gold_heads)
            return active_heads, xs_ls, metrics
        else:
            return None, None, metrics

    def reset_metrics(self):
        self.test_metrics = Metrics()
