from collections import OrderedDict
import copy
import numpy as np
import ast
import operator
from ..model.rule import RuleGrounding
from ..model.label import LabelType
from inferencer_base import InferencerBase

class ViterbiInferencer(InferencerBase):
    def __init__(self, ruleGroundingsInput, constraintGroundingsInput):
        super(ViterbiInferencer, self).__init__(ruleGroundingsInput, constraintGroundingsInput)
        self.cumScore = {}
        self.backptrs = {}
        self.preds = {}
        self.golds = {}

    def encode(self):
        #TO-DO: any preprocessing of rules goes here
		pass

    def optimize(self):
        #TO-DO: HasLabel can't be hardcoded
        i = 0
        for rg in self.ruleGroundings.keys():
            i += 1
            score = self.ruleGroundings[rg]
            head_args = rg.head['arguments']
            body_args = []
            for el in rg.get_body_predicates("HasLabel"):
                body_args = el["arguments"]
            if head_args[0] not in self.cumScore.keys():
                self.cumScore[head_args[0]] = {}
                self.backptrs[head_args[0]] = {}
            if body_args[0]==-1:
                self.cumScore[head_args[0]][head_args[1]] = score
                self.backptrs[head_args[0]][head_args[1]] = [-1,-1]
            else:
                if head_args[1] not in self.cumScore[head_args[0]]:
                    self.cumScore[head_args[0]][head_args[1]] = 0
                newCumScore = score + self.cumScore[body_args[0]][body_args[1]]
                if newCumScore > self.cumScore[head_args[0]][head_args[1]]:
                    self.cumScore[head_args[0]][head_args[1]] = newCumScore
                    self.backptrs[head_args[0]][head_args[1]] = body_args
            if i==len(self.ruleGroundings.keys()):
                cumScore = self.cumScore[head_args[0]]
                val = max(cumScore.iteritems(),key=operator.itemgetter(1))[0]
                backptr = self.backptrs[head_args[0]][val]
                self.preds[head_args[0]] = val
                while 1:
                    if backptr[0] == -1:
                        break
                    self.preds[backptr[0]] = backptr[1]
                    #print backptr[0],' should be ',backptr[1]
                    backptr = self.backptrs[backptr[0]][backptr[1]]

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
        multiclass_metrics = self.evaluate_multiclass(gold_heads, binary_predicates)
        return multiclass_metrics

