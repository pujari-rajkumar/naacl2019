import sys
import numpy as np
import torch
from collections import OrderedDict
from collections import Counter
import json
from importlib import import_module
import imp
import os

from metrics import Metrics
from ..database import Database
from ..parser import parser
from ..features.feature_extractor import FeatureExtractor
from ..model.rule import RuleGrounding
from ..model.scoring_function import ScoringType
from ..neuro import nn_utils

class Learner(object):

    def __init__(self):
        self.par = parser.Parser()
        # will be created during method calls
        self.ruleset = None
        self.group_by = None
        self.reset()

    def reset(self):
        self.nnindex_ls = {}
        self.fe = None
        self.binary_classifiers = set([])
        torch.manual_seed(12345)
        torch.cuda.manual_seed(12345)

    def compile_rules(self, rule_path):
        self.par.build()
        self.par.parse(rule_path)
        self.ruleset = self.par.rulesets[0]
        self.group_by = self.par.groupby[0]

    def create_dataset(self, dataset_path, **kwargs):
        if self.par.dbmodule is None:
            db = Database()
            db.load_predicates(self.par.predicate_arguments,
                               dataset_path, self.par.files)
            db.load_predicates(self.par.entity_arguments,
                               dataset_path, self.par.files)
            db.load_labels(self.par.label_types.keys(),
                           dataset_path, self.par.files)
            db.predicate_entities = self.par.predicate_entities

        else:
            dbmodule_path = kwargs.pop('dbmodule_path', '.')
            module_path = os.path.join(dbmodule_path, "{0}.py".format(self.par.dbmodule))
            mod = imp.load_source(self.par.dbmodule, module_path)
            db_class = getattr(mod, self.par.dbclass)
            db = db_class()
            db.load_data(dataset_path)
        return db

    def _get_multiclass_train(self, db, filters, ruleidx, rule_template, split_class):
        if self.par.dbmodule is None:
            train_groundings = db.unfold_train_groundings(
                    rule_template, ruleidx, split_class,
                    filter_by=filters)
        else:
            train_groundings = getattr(db, rule_template.dbfunc)(
                    istrain=True,
                    isneg=False,
                    filters=filters,
                    split_class=None,
                    instance_id=None)
        return train_groundings

    def _get_binary_train(self, db, filters, ruleidx, rule_template, split_class):
        if self.par.dbmodule is None:
            neg_train_groundings = db.unfold_train_groundings(
                    rule_template, ruleidx, split_class,
                    filter_by=filters, neg_head=True)
            pos_train_groundigs = db.unfold_train_groundings(
                    rule_template, ruleidx, None,
                    filter_by=filters, neg_head=False)
        else:
            neg_train_groundings = \
                    getattr(db, rule_template.dbfunc)(istrain=True,
                          isneg=True,
                          filters=filters,
                          split_class=split_class,
                          instance_id=None)
            pos_train_groundings = \
                    getattr(db, rule_template.dbfunc)(istrain=True,
                          isneg=False,
                          filters=filters,
                          split_class=split_class,
                          instance_id=None)
            return neg_train_groundings, pos_train_groundings

    def build_feature_extractors(self, db, filters=[], **kwargs):
        femodule_path = kwargs.pop('femodule_path', '.')
        module_path = os.path.join(femodule_path, "{0}.py".format(self.par.femodule))
        mod = imp.load_source(self.par.femodule, module_path)
        fe_class = getattr(mod, self.par.feclass)
        self.fe = fe_class(**kwargs)
        self.fe.build()

        for ruleidx, rule_template in enumerate(self.ruleset['rule']):
            print rule_template
            # we don't need a NNet or feat extractor for constraint rules
            #if rule_template.isconstr:
            #    continue

            if rule_template.scoring_function.sctype == ScoringType.NNetRef:
                scref = rule_template.scoring_function.scref
                rule_template.feat_vector_sz = self.ruleset['rule'][scref].feat_vector_sz
            elif rule_template.scoring_function.sctype == ScoringType.NNet:
                if rule_template.head.isobs or rule_template.split_on is not None:
                    # multiclass
                    train_groundings = \
                            self._get_multiclass_train(db, filters, ruleidx,
                                    rule_template, split_class=None)
                else:
                    # binary
                    neg_train_groundings, pos_train_groundings = \
                            self._get_binary_train(db, filters, ruleidx,
                                    rule_template, split_class=None)
                    train_groundings = neg_train_groundings + pos_train_groundings

                print "Train groundings", len(train_groundings)
                print rule_template.feat_functions
                feats = self.fe.extract(train_groundings[0], rule_template.feat_functions)
                rule_template.feat_vector_sz = len(feats['vector'])
                rule_template.feat_inputs_sz = [len(inp) for inp in feats['input']]
            # in the scoring function case I don't care about feature vector size

    def _build_model(self, rule_template, ruleidx, configs,
                     nn_index, nns_ls, shared_layers):
        output_dim = rule_template.head_cardinality()
        config = configs["models"][ruleidx]
        use_gpu = config['use_gpu']
        module = "drail.neuro.{0}".format(config['module'])
        mod = import_module(module)
        neuro_class = getattr(mod, config['model'])
        nn = neuro_class(config, nn_index, use_gpu, output_dim)
        if use_gpu:
            nn.cuda()


        nn.build_architecture(rule_template, self.fe, shared_layers)
        nns_ls.append(nn)

    def _create_shared_layers(self, config):
        shared_layers = {}
        if "shared_layers" in config:
            for shared in config["shared_layers"]:
                shared_layers[shared] = {}
                shared_layers[shared]['layer'] = \
                        nn_utils.create_layer(config["shared_layers"][shared])
                shared_layers[shared]['nin'] = \
                        config["shared_layers"][shared]["nin"]
                shared_layers[shared]['nout'] = \
                        config["shared_layers"][shared]["nout"]
        return shared_layers

    def _build_models(self, db, config_path, nns_ls, isdic=False):
        if not isdic:
            with open(config_path, 'r') as f:
                configs = json.load(f)
        else:
            configs = config_path

        shared_layers = self._create_shared_layers(configs)

        nn_index = 0
        for ruleidx, rule_template in enumerate(self.ruleset['rule']):
            print rule_template

            # check if we need a neural network for this rule
            if rule_template.scoring_function.sctype != ScoringType.NNet:
                continue

            rule_template.scoring_function.scref = ruleidx

            # obtain classifier split if it exists
            if rule_template.split_on is not None:
                classes = db.get_distinct_values(rule_template.head, rule_template.split_on)
                rule_template.split_on_classes = [class_[0] for class_ in classes]

            template_nnets = []
            if len(rule_template.split_on_classes) > 0:
                for class_ in rule_template.split_on_classes:
                    print "Building nnet for predicate {0} class {1}".format(
                            rule_template.head.name, class_)
                    self._build_model(rule_template, ruleidx, configs, nn_index, nn_ls,
                                      shared_layers)
                    template_nnets.append(nn_index)
                    nn_index += 1
                    self.test_metrics.add_classifier(rule_template.head.name)

                    if rule_template.split_on_ttype == LabelType.Binary:
                        self.binary_classifiers.add(rule_template.head.name)
            else:
                print "Building nnet for predicate {0}".format(
                        rule_template.head.name)

                nn = self._build_model(rule_template, ruleidx, configs, nn_index, nns_ls,
                                       shared_layers)
                template_nnets.append(nn_index)
                nn_index += 1
                self.test_metrics.add_classifier(rule_template.head.name)

                if not rule_template.head.isobs:
                    self.binary_classifiers.add(rule_template.head.name)

            self.nnindex_ls[ruleidx] = template_nnets

    def _get_groundings(self, db, filters, ruleidx, rule_template,
                       inst, class_split, gold_heads):
        if self.par.dbmodule is None:
            gold_heads_set = set([RuleGrounding.pred_str(head) for head in gold_heads])
            rule_groundings = db.unfold_rule_groundings(
                    rule_template, ruleidx, self.group_by,
                    inst, class_split, gold_heads_set, filter_by=filters)
        else:
            rule_groundings = \
                getattr(db, rule_template.dbfunc)(
                        istrain=False,
                        isneg=False,
                        filters=filters,
                        split_class=class_split,
                        instance_id=inst)
        return rule_groundings

    def _get_instance_groundings(self, db, ruleidx, rule_template,
                                 filters, class_split, gold_heads, inst):
        dic = OrderedDict()
        rule_groundings = self._get_groundings(
            db, filters, ruleidx, rule_template, inst, class_split, gold_heads)
        '''
        for rg in rule_groundings:
            print rg
        '''
        for rg in rule_groundings:
            # create a unique repr for the body
            body_str = ' & '.join([RuleGrounding.pred_str(predicate) for predicate
                        in rg.body])

            # extract heads in left hand side
            unobs = set([RuleGrounding.pred_str(predicate) for predicate in rg.body if
                        not predicate['obs']])

            # extract current head and indexes
            head = RuleGrounding.pred_str(rg.head)

            if not rg.is_binary_head:
                head_index = self.fe.extract_multiclass_head(rg)
            else:
                head_index = None

            # if we havent seen this body before
            if body_str not in dic:
                # extract features (only once per body)
                rule_grd_x = self.fe.extract(rg, rule_template.feat_functions)
                lmbd = 1
                dic[body_str] = \
                    {'groundings': [rg], 'feat_repr': rule_grd_x,
                      'heads': [head], 'heads_index': [head_index],
                      'unobs': unobs, 'lambda': lmbd}
            else:
                # append info if we have seen this body before
                dic[body_str]['groundings'].append(rg)
                dic[body_str]['heads_index'].append(head_index)
                dic[body_str]['heads'].append(head)

        return dic

    def extract_rules(self, db, filters, is_global_train=False):
        print "Extracting rules..."
        instances = db.get_ruleset_instances(self.ruleset['rule'], self.group_by,
                filters, is_global_train)
        #instances = set(list(instances)[:10])
        num = len(instances)

        instance_groundings = {}; gold_heads_ret = {}
        # extraction of data
        for i, inst in enumerate(instances):
            #print i, inst
            gold_heads = db.get_gold_predicates(self.ruleset['rule'], filters,
                                                self.group_by, inst)
            gold_heads_ret[inst] = gold_heads
            instance_groundings[inst] = []

            for ruleidx, rule_template in enumerate(self.ruleset['rule']):
                #print
                #print "\t", ruleidx, rule_template
                instance_groundings[inst].append([])

                # if associated to a neural network, take care of subnet splits
                scref = rule_template.scoring_function.scref
                if scref is not None:
                    for index, nnidx in enumerate(self.nnindex_ls[scref]):
                        class_split = None
                        if len(rule_template.split_on_classes) > 0:
                            class_split = rule_template.split_on_classes[index]

                        dic = self._get_instance_groundings(
                            db, ruleidx, rule_template, filters, class_split,
                            gold_heads, inst, )
                        instance_groundings[inst][ruleidx].append(dic)
                        #print "\t\t", len(dic)
                else:
                    class_split = None
                    dic = self._get_instance_groundings(
                        db, ruleidx, rule_template, filters, class_split,
                        gold_heads, inst)
                    instance_groundings[inst][ruleidx].append(dic)
            #sys.stdout.write("\r\t{0}%".format((i * 100) / num))
            '''
            if i == 1:
                exit()
            '''
        #sys.stdout.write('\n')
        print "Done"
        return instances, instance_groundings, gold_heads_ret

    def extract_constraints(self, db, filters, is_global_train=False):
        # obtain inference instances
        print "Extracting constraints..."
        instances = db.get_ruleset_instances(self.ruleset['rule'], self.group_by,
                                             filters, is_global_train)
        num = len(instances)
        print("Num constraints: {0}".format(num))
        constraint_groundings = {}; constraint_heads_ret = {}

        for i, inst in enumerate(instances):
            #print i, inst
            constraint_heads_ret[inst] = set([])
            constraint_groundings[inst] = []
            # extract constraints for the instance
            if 'constr' in self.ruleset:
                for constidx, constr_template in enumerate(self.ruleset['constr']):
                    #print "\t", constidx, constr_template
                    if self.par.dbmodule is None:
                        cg = db.unfold_rule_groundings(
                                constr_template, constidx, self.group_by, inst,
                                    None, set([]), isconstr=True, filter_by=filters)
                    else:
                        cg = getattr(db, constr_template.dbfunc)(filters, inst)
                    #for gr in cg:
                    #    print "\t\t", gr
                    #print "\t\t", len(cg)
                    constraint_groundings[inst] += cg
                    # add enforced head constraints to gold dataset
                    n_unobs = sum(1 for pred in constr_template.body if not pred.isobs)
                    if n_unobs == 0:
                        for cg in constraint_groundings[inst]:
                            if (cg.head['ttype'] != 1):
                                constraint_heads_ret[inst].add(RuleGrounding.pred_str(cg.head))
			#sys.stdout.write("\r\t{0}%".format((i * 100) / num))
        #sys.stdout.write('\n')
        print "Done"
        return constraint_groundings, constraint_heads_ret


    def _add_features(self, feats, X):
        X['vector'].append(feats['vector'])
        X['input'].append(feats['input'])
        for emb in feats['embedding']:
            if emb not in X['embedding']:
                X['embedding'][emb] = []
            X['embedding'][emb].append(feats['embedding'][emb])
        return X

    def _extract_features(self, ruleidx, rule_template, observed_groundings, Y=[]):
        X = {'vector': [], 'embedding': {}, 'input': []}
        for rg in observed_groundings:
            feats = self.fe.extract(rg, rule_template.feat_functions)
            X = self._add_features(feats, X)
            if rule_template.head.isobs:
                head_index = self.fe.extract_multiclass_head(rg)
                Y.append(head_index)
        return X, Y

    def _observed_data(self, db, rule_template, ruleidx,
            filters, split_class, limit=-1):

        Y = []
        if rule_template.head.isobs:
            # multiclass
            observed_groundings = self._get_multiclass_train(
                    db, filters, ruleidx, rule_template, split_class)
            if limit > 0:
                shuffle(observed_groundings)
                observed_groundings = observed_groundings[:limit]
        else:
            # binary
            neg_obs_groundings, pos_obs_groundings = \
                    self._get_binary_train(db, filters, ruleidx,
                        rule_template, split_class)
            if limit > 0 and len(neg_obs_groundings) > limit:
                shuffle(neg_obs_groundings)
                neg_obs_groundings = neg_obs_groundings[:limit]
            if limit > 0 and len(pos_obs_groundings) > limit:
                shuffle(pos_obs_groundings)
                pos_obs_groundings = pos_obs_groundings[:limit]

            observed_groundings = neg_obs_groundings + pos_obs_groundings
            Y = [0] * len(neg_obs_groundings) + [1] * len(pos_obs_groundings)
            print "neg:", len(neg_obs_groundings), "pos:", len(pos_obs_groundings)

        X, Y = self._extract_features(ruleidx, rule_template, observed_groundings, Y)
        return X, Y

    def get_observed_data(self, db, fold_filters):
        observed= {};
        for ruleidx, rule_template in enumerate(self.ruleset['rule']):
            #print rule_template

            # if no training will be done we don't need to get observed data
            if rule_template.scoring_function.sctype != ScoringType.NNet:
                continue

            observed[ruleidx] = []
            for index, nnidx in enumerate(self.nnindex_ls[ruleidx]):

                if len(rule_template.split_on_classes) == 0:
                    class_split = None
                else:
                    class_split = rule_template.split_on_classes[index]

                X, Y = self._observed_data(db, rule_template, ruleidx,
                                          fold_filters, class_split)

                observed[ruleidx].append((X, Y))
        return observed

