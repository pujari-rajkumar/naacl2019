from collections import OrderedDict
import copy
import numpy as np
import ast
import heapq
import operator
from ..model.rule import RuleGrounding
from ..model.label import LabelType
from inferencer_base import InferencerBase

class BeamInferencer(InferencerBase):

    def __init__(self, ruleGroundingsInput, constraintGroundingsInput):
        super(BeamInferencer, self).__init__(ruleGroundingsInput, constraintGroundingsInput)
        self.rules = {}
        self.preds = {}
        self.prev_choice = []
        # TO-DO: be able to pass as arg and define in DRaiL script
        self.beam_width = 2

    def encode(self):
        # TO-DO: HasLabel can't be hardcoded
        for rg in self.ruleGroundings.keys():
            body_args = rg.get_body_predicates("HasLabel")[0]["arguments"]
            head_arg = rg.head["arguments"]
            if body_args[0] not in self.rules:
                self.rules[body_args[0]] = {}
            if body_args[1] not in self.rules[body_args[0]]:
                self.rules[body_args[0]][body_args[1]] = {}
            if head_arg[0] not in self.rules[body_args[0]][body_args[1]]:
                self.rules[body_args[0]][body_args[1]][head_arg[0]] = {}
            if head_arg[1] not in self.rules[body_args[0]][body_args[1]][head_arg[0]]:
                self.rules[body_args[0]][body_args[1]][head_arg[0]][head_arg[1]] = self.ruleGroundings[rg]
        #print self.rules

    def optimize(self):
        initial = self.rules[-1]
        key = initial['*'].keys()[0]
        reqd_dict = initial['*'][key]
        backptrs = {}
        prev_word = key
        self.prev_choice = heapq.nlargest(self.beam_width, initial['*'][key],key=initial['*'][key].__getitem__)
        reqd_dict = { your_key: reqd_dict[your_key] for your_key in self.prev_choice }
        backptrs[key] = {}
        for choice in self.prev_choice:
            backptrs[key][choice] = [-1,'*']
        #print reqd_dict
        while(1):
            cands = {}
            for choice in reqd_dict:
                key = self.rules[prev_word][choice].keys()[0]
                n_word = key
                new_reqd_dict = self.rules[prev_word][choice][key]
                new_reqd_dict.update((x, y+reqd_dict[choice]) for x, y in new_reqd_dict.items())
                if len(cands)==0:
                    cands = new_reqd_dict
                    for k in cands:
                        if key not in backptrs:
                            backptrs[key] = {}
                        backptrs[key][k] = [prev_word,choice]
                else:
                    for c in new_reqd_dict:
                        if cands[c]<new_reqd_dict[c]:
                            cands[c] = new_reqd_dict[c]
                            backptrs[key][c] = [prev_word,choice]
            prev_word = n_word
            self.prev_choice = heapq.nlargest(self.beam_width, cands,key=cands.__getitem__)
            reqd_dict = { your_key: cands[your_key] for your_key in self.prev_choice }

            if prev_word not in self.rules:
                #Now find the last highest score value and backtrack
                k = max(reqd_dict.iteritems(), key=operator.itemgetter(1))[0]
                self.preds[prev_word] = k
                last = backptrs[prev_word][k]
                while last[0]!=-1:
                    self.preds[last[0]] = last[1]
                    last = backptrs[last[0]][last[1]]
                break

    def evaluate_multiclass(self, gold_heads, binary_predicates=[]):
        metrics = {}

        # track gold assignments
        gold_data = {}
        for elem in gold_heads:
            classif_name = elem['name']

            if classif_name in binary_predicates:
                continue

            if classif_name not in gold_data:
                gold_data[classif_name] = {}
            id_ = elem['arguments'][0]
            label = elem['arguments'][1]
            gold_data[classif_name][id_] = label

        # track pred assignments
        pred_data = {}
        for head in gold_heads:
            # skip things we are not evaluating on
            if head['name'] not in gold_data:
                continue

            if head['name'] not in metrics:
                metrics[head['name']] = {'gold_data': [], 'pred_data': []}
                pred_data[head['name']] = {}

            curr_id = head['arguments'][0]
            k_class = self.preds[curr_id]
            pred_data[head['name']][curr_id] = k_class

        # create ordered list of predictions to be used in sklearn
        for classif_name in gold_data:

            if len(pred_data[classif_name]) != len(gold_data[classif_name]):
                continue

            for id_ in gold_data[classif_name]:
                gold_label = gold_data[classif_name][id_]
                pred_label = pred_data[classif_name][id_]
                metrics[classif_name]['gold_data'].append(gold_label)
                metrics[classif_name]['pred_data'].append(pred_label)

        return metrics

    def evaluate(self,gold_heads, binary_predicates=[]):
        # TO-DO: add binary / multilabel evaluation
        multiclass_metrics = self.evaluate_multiclass(gold_heads, binary_predicates)
        return multiclass_metrics
