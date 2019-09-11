import numpy as np
import os
import json
import sys
from collections import OrderedDict
import os
import traceback
from metrics import Metrics
from random import shuffle
from sklearn.metrics import *
from sklearn.preprocessing import scale
from importlib import import_module
import imp
import random

from learner import Learner
from ..neuro import nn_utils, nn_model
from ..neuro.nn_feedforward import FeedForwardNetwork
from ..neuro.nn_bilstm import BiLSTM
from ..inference.ilp_inferencer import *
from ..inference.viterbi_inferencer import *
from ..inference.beam_inferencer import *
from ..model.rule import RuleGrounding
from ..model.label import LabelType
from ..model.scoring_function import ScoringType

class LocalLearner(Learner):

    def __init__(self, gpu=False):
        super(LocalLearner, self).__init__()
        self.reset()
        random.seed(2017)

    def reset(self):
        super(LocalLearner, self).reset()
        self.observed_predicates = set([])
        self.train_metrics = Metrics()
        self.dev_metrics = Metrics()
        self.test_metrics = Metrics()

    def build_models(self, db, config_path, isdic=False):
        self.nns_ls = []
        self._build_models(db, config_path, self.nns_ls, isdic)

    def extract_data(self, db, test_filters=[], dev_filters=[], extract_dev=False):
        # extract test data once
        print "Extracting test rules..."
        self.instances = {}; self.gold_heads = {}
        self.instance_groundings = {}; self.constraint_groundings = {}

        self.instances['test'], self.instance_groundings['test'],\
        self.gold_heads['test'] = \
                self.extract_rules(db, test_filters)
        self.constraint_groundings['test'], _ = \
                self.extract_constraints(db, test_filters)

        if extract_dev:
            self.instances['dev'], self.instance_groundings['dev'],\
            self.gold_heads['dev'] = \
                    self.extract_rules(db, dev_filters)
            self.constraint_groundings['dev'], _ = \
                    self.extract_constraints(db, dev_filters)


    def train(self, db, train_filters=[], dev_filters=[], test_filters=[], limit=-1):
        perf = {}
        instance_map = {}

        # extract training data
        print "extract train..."
        observed_train = self.get_observed_data(
                db, train_filters)
        print "extract dev..."
        observed_dev = self.get_observed_data(
                db, dev_filters)
        print "extract test..."
        observed_test = self.get_observed_data(
                db,  test_filters)

        for ruleidx, rule_template in enumerate(self.ruleset['rule']):
            if rule_template.scoring_function.sctype != ScoringType.NNet:
                continue

            print '\nTraining/Testing for rule #%d : \n'%(ruleidx), rule_template

            perf[rule_template]={}
            for index, nnidx in enumerate(self.nnindex_ls[ruleidx]):
                nn = self.nns_ls[nnidx]

                X_train, Y_train = observed_train[ruleidx][index]
                X_dev, Y_dev = observed_dev[ruleidx][index]
                X_test, Y_test = observed_test[ruleidx][index]

                class_split = None
                if len(rule_template.split_on_classes) > 0:
                    class_split = rule_template.split_on_classes[index]

                print len(Y_train), len(Y_dev), len(Y_test)

                perf[rule_template][class_split] = \
                        [nn_utils.train_local_on_epoches(nn,
                            X_train, Y_train, X_dev, Y_dev, X_test, Y_test),\
                         nn]

        return perf


    def _load_weights(self, instance_groundings, y_pred, pivots, inference_weights, lmbd):
        # instantiate weights
        for body, scores, pivot in zip(instance_groundings, y_pred, pivots):
            for gr_i, gr in enumerate(instance_groundings[body]['groundings']):
                if gr.is_binary_head:
                    idx_in_nn_out = 1
                else:
                    idx_in_nn_out = instance_groundings[body]['heads_index'][gr_i]
                weight = 0.0
                if idx_in_nn_out < len(scores):
                    weight = scores[idx_in_nn_out]
                # if weight not in the top K solutions, do not add it to the problem
                if weight >= pivot:
                    inference_weights[gr] = weight * lmbd

    def _score_neural(self, scref, instance_groundings, inference_weights, K, lambda_):
        for index, nnidx in enumerate(self.nnindex_ls[scref]):
            nn = self.nns_ls[nnidx]
            x = {'vector': [], 'embedding': {}, 'input': []};
            y_pred = []

            if len(instance_groundings[index].keys()) > 0:

                # predict with neural nets
                for body in instance_groundings[index]:
                    x = self._add_features(
                            instance_groundings[index][body]['feat_repr'],
                            x)

                y_pred = nn_utils.predict_local_scores(nn, x)
                #print "y_pred (score_neural) ", y_pred.shape
                partial_order = np.partition(y_pred, y_pred.shape[1] - K)
                pivots = partial_order[:,y_pred.shape[1] - K]

                self._load_weights(instance_groundings[index], y_pred, pivots,
                                   inference_weights, lambda_)

    def _score_manual(self, scmodule, scoring_func, instance_groundings,
                      inference_weights, K, lambda_):
        x = {'vector': [], 'embedding': {}, 'input': []};
        y_pred = []
        index = 0
        if len(instance_groundings[index].keys()) > 0:

            # predict with neural nets
            for body in instance_groundings[index]:
                x = self._add_features(
                        instance_groundings[index][body]['feat_repr'],
                        x)
            y_pred = getattr(scmodule, scoring_func)(x, self.fe)
            partial_order = np.partition(y_pred, y_pred.shape[1] - K)
            pivots = partial_order[:,y_pred.shape[1] - K]

            self._load_weights(instance_groundings[index], y_pred, pivots,
                               inference_weights, lambda_)

    # TO-DO: set a K per predicate, can't be a single one
    def _instance_scores(self, fold, inst, K=None, scmodule_path=None, lmb_weights=None):
        # load module if available
        if scmodule_path is not None:
            module_path = os.path.join(scmodule_path, "{0}.py".format(self.par.scmodule))
            scmodule = imp.load_source(self.par.scmodule, module_path)

        inference_weights = OrderedDict()
        for ruleidx, rule_template in enumerate(self.ruleset['rule']):
            if lmb_weights is not None:
                lmbd_weight = lmb_weights[ruleidx]
            else:
                lmbd_weight = 1.0


            #print "ruleidx", ruleidx, "lmbd_weight", lmbd_weight

            if K is None:
                K_curr = rule_template.head_cardinality()

            scref = rule_template.scoring_function.scref

            if scref is not None:
                self._score_neural(scref, self.instance_groundings[fold][inst][ruleidx],
                                   inference_weights, K_curr, lmbd_weight * rule_template.lambda_)
            else:
                # TO-DO: retrieve scoring function from script and execute
                self._score_manual(scmodule, rule_template.scoring_function.scfunc,
                                   self.instance_groundings[fold][inst][ruleidx],
                                   inference_weights, K_curr, lmbd_weight * rule_template.lambda_)

        return inference_weights

    def _run_inference(self, fold, inst, inference_weights, get_active_heads=False):
        '''
        for const in self.constraint_groundings[fold][inst]:
            print "const", const
        '''
        inferencer = ILPInferencer(inference_weights, self.constraint_groundings[fold][inst])
        inferencer.encode()
        inferencer.optimize()

        seed_heads = [RuleGrounding.pred_str(cnstr.head) for cnstr in self.constraint_groundings[fold][inst]\
                      if sum(1 for pred in cnstr.body if not pred['obs']) == 0]
        seed_heads = set(seed_heads)

        metrics = inferencer.evaluate(self.gold_heads[fold][inst], binary_predicates=self.binary_classifiers)

        active_heads = None
        if get_active_heads:
            active_heads = inferencer.get_predictions()
        return metrics, active_heads


    def _init_lmbd_weights(self):
        weights = {}
        for ruleidx, rule_template in enumerate(self.ruleset['rule']):
            #weights[ruleidx] = random.uniform(0, 0.0025)
            weights[ruleidx] = 1.0
        #weights[0] = 100000.0
        return weights

    def _rule_lambdas(self, rule_template, bgs, active_heads):
        lambdas = 0
        for b_index, body in enumerate(bgs):
            if len(bgs[body]['unobs'] & active_heads) == len(bgs[body]['unobs']):
                if not rule_template.head.isobs and \
                   bgs[body]['heads'][0] in active_heads:
                    phi[ruleidx] += rule_template.lambda_
                else:
                    for head, head_index in zip(bgs[body]['heads'], bgs[body]['heads_index']):
                        if head in active_heads:
                           lambdas += rule_template.lambda_
        return lambdas

    def _create_lambdas(self, instance_groundings, active_heads):
        phi = {}
        for ruleidx, rule_template in enumerate(self.ruleset['rule']):
            phi[ruleidx] = 0
            scref = rule_template.scoring_function.scref
            if scref is not None:
                for index, nnidx in enumerate(self.nnindex_ls[scref]):
                    bgs = instance_groundings[ruleidx][index]
                    phi[ruleidx] += self._rule_lambdas(rule_template, bgs, active_heads)
            else:
                bgs = instance_groundings[ruleidx][0]
                phi[ruleidx] += self._rule_lambdas(rule_template, bgs, active_heads)
        return phi

    def _update_lmbd_weights(self, weights, phi_prime, phi_hat, lr):
        for ruleidx, rule_template in enumerate(self.ruleset['rule']):
            weights[ruleidx] += lr * (phi_prime[ruleidx] - phi_hat[ruleidx])

    def _accuracy(self, metrics, predicates=[]):
        accuracy = {}
        y_gold_all = []; y_pred_all = []
        for pred in predicates:
            y_gold = metrics.metrics[pred]['gold_data']
            y_pred = metrics.metrics[pred]['pred_data']

            y_gold_all += y_gold
            y_pred_all += y_pred

            acc_test = accuracy_score(y_gold, y_pred)
            accuracy[pred] = acc_test
        acc_test = accuracy_score(y_gold_all, y_pred_all)
        accuracy['all'] = acc_test
        return accuracy

    # Train lambdas on specified fold
    def train_lambdas(self, iterations, lr, fold, K=None, scmodule_path=None,
                      predicates=[]):
        weights = self._init_lmbd_weights()

        for _iter in range(iterations):
            dev_metrics = Metrics()
            phi_prime = {}; phi_hat = {}
            for i, inst in enumerate(self.instances[fold]):
                inference_weights = self._instance_scores(fold, inst, K, scmodule_path, weights)

                _, active_heads = self._run_inference(fold, inst, inference_weights,
                                                            get_active_heads=True)
                gold_heads = set([RuleGrounding.pred_str(h) for h in self.gold_heads[fold][inst]])

                # correct predictions
                res = self._create_lambdas(self.instance_groundings[fold][inst], active_heads & gold_heads)
                for ruleidx in res:
                    if ruleidx not in phi_prime:
                        phi_prime[ruleidx] = 0.0
                    phi_prime[ruleidx] += res[ruleidx]

                # incorrect predictions
                res = self._create_lambdas(self.instance_groundings[fold][inst], active_heads - gold_heads)
                for ruleidx in res:
                    if ruleidx not in phi_hat:
                        phi_hat[ruleidx] = 0.0
                    phi_hat[ruleidx] += res[ruleidx]

            self._update_lmbd_weights(weights, phi_prime, phi_hat, lr)

            print "weights", weights
            for i, inst in enumerate(self.instances[fold]):
                inference_weights = self._instance_scores(fold, inst, K, scmodule_path, weights)
                metrics, _ = self._run_inference(fold, inst, inference_weights)
                dev_metrics.load_metrics(metrics)

            print _iter, self._accuracy(dev_metrics, predicates)
            print
        return weights

    def predict_local_topK(self, rules, ridx, index, nnindex, K=10):
        inference_weights = OrderedDict()
        rule_template = self.ruleset['rule'][ridx]

        for rule in rules:
            nn = self.nns_ls[nnindex]
            x = {'vector': [], 'embedding': {}, 'input': []};
            y_pred = []

            feats = self.fe.extract(rule, rule_template.feat_functions)
            x = self._add_features(feats, x)
            y_pred = nn_utils.predict_local_scores(nn, x)
            # WARNING: HARD CODED FOR BINARY
            weight = y_pred[0][1]
            inference_weights[rule] = weight
            print rule, weight
        return inference_weights

    def predict(self, db, fold='test', K=None, scmodule_path=None, lmbd_weights=None,
                get_predicates=False):
        print "predicting..."
        num = len(self.instances[fold])
        test_metrics = Metrics()

        all_predictions = set([])
        for i, inst in enumerate(self.instances[fold]):
            #print
            #print i, inst
            inference_weights = \
                self._instance_scores(fold, inst, K, scmodule_path, lmbd_weights)
            '''
            for gr in inference_weights:
                print gr, inference_weights[gr]
            '''
            metrics, active_heads = self._run_inference(fold, inst, inference_weights,
                                                        get_active_heads=get_predicates)
            #print metrics
	    if active_heads:
                all_predictions |= active_heads
            test_metrics.load_metrics(metrics)
            #exit()
        if get_predicates:
            return test_metrics, all_predictions
        else:
            return test_metrics

    def get_test_rule_templates(self):
        return self.ruleset['rule']

    def reset_metrics(self):
        self.test_metrics = Metrics()
