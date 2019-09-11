from collections import OrderedDict
import copy
import gurobipy as grb
import numpy as np
import ast

from ..model.rule import RuleGrounding
from ..model.label import LabelType
from inferencer_base import InferencerBase

class ILPInferencer(InferencerBase):

    def __init__(self, ruleGroundingsInput, constraintGroundingsInput):
        super(ILPInferencer, self).__init__(ruleGroundingsInput, constraintGroundingsInput)
        """
        Initializes a Gurobi ILP Inferencer given a set of rule groundings.

        Args:
            ruleGroundingsInput: a list of rule groundings
            relaxation: optimization relaxation ('MIP', 'ILP', 'LP')
        """
        self.model = None
        self.relaxation = 'ILP'
        #print("Finished initialization")
        #print("Updated rule weights")
        #print("# of rule groundings is {}".format(len(self.ruleGroundings)))

    def createModel(self, modelName, timeLimit = None):
        """
        Creates the Gurobi model.

        Args:
            modelName: model name, string type
        """
        self.model = grb.Model(modelName)
        self.model.setParam('OutputFlag', 0)

        if timeLimit is not None:
            self.model.setParam('TimeLimit', timeLimit)
		# Set the sense to maximization
        self.model.setAttr("ModelSense", -1)
        #print("Created model")

    def addVariables(self):
        """
        Adds variables based on all rule groundings in this instance.
        TO-DO: modularize code for readability
        """
        #print ("Adding Variables...")
        self.headDict = OrderedDict()
        self.multiClassDict = OrderedDict()
        self.multiLabelDict = OrderedDict()
        self.binaryDict = OrderedDict()
        self.numRules = 0

        ruleVarCounter = 0
        headCounter = 0
        debugout = open('ruleWeights.txt', 'w')

        for ruleGrounding in self.ruleGroundings.keys():
            debugout.write(str(ruleGrounding) + ' ' +
                    str(self.ruleGroundings[ruleGrounding]) + ' ' +
                    str(ruleVarCounter) + '\n')

            if self.relaxation != 'ILP':
                rvar = self.model.addVar(
                        obj=self.ruleGroundings[ruleGrounding],
                        lb = 0.0, ub = 1.0, vtype=grb.GRB.CONTINUOUS,
                        name='r_'+str(ruleVarCounter))
            else:
                rvar = self.model.addVar(
                        obj=self.ruleGroundings[ruleGrounding],
                        lb = 0.0, ub = 1.0, vtype=grb.GRB.BINARY,
                        name='r_'+str(ruleVarCounter))


            ruleVarCounter += 1
            # print "weight: {}".format(self.ruleGroundings[ruleGrounding])

            # BinaryPredicate case
            if (ruleGrounding.is_binary_head and 
                    (ruleGrounding.head['target_pos'] is None or
                     ruleGrounding.head['ttype'] == LabelType.Binary)
                ):

                if self.relaxation != 'ILP':
                    rvar2 = self.model.addVar(
                            obj=1-self.ruleGroundings[ruleGrounding],
                            lb= 0.0, ub = 1.0, vtype=grb.GRB.CONTINUOUS,
                            name='r_'+str(ruleVarCounter))
                else:
                    rvar2 = self.model.addVar(
                            obj=1-self.ruleGroundings[ruleGrounding],
                            lb= 0.0, ub = 1.0, vtype=grb.GRB.BINARY,
                            name='r_'+str(ruleVarCounter))
                ruleVarCounter += 1

            head = ruleGrounding.head
            negation = copy.deepcopy(ruleGrounding.head)
            negation['isneg'] = not negation['isneg']

            if (RuleGrounding.pred_str(head) not in self.headDict and
                    RuleGrounding.pred_str(negation) not in self.headDict):

                if self.relaxation == 'LP':
                    headVar = self.model.addVar(obj=0, lb = 0.0, ub = 1.0,
                            vtype=grb.GRB.CONTINUOUS,
                            name='h_'+str(headCounter))
                else:
                    headVar = self.model.addVar(obj=0, lb = 0.0, ub = 1.0,
                            vtype=grb.GRB.BINARY,
                            name='h_'+str(headCounter))
                headCounter += 1
                if self.relaxation == 'LP':
                    negVar = self.model.addVar(obj=0, lb = 0.0, ub = 1.0,
                            vtype=grb.GRB.CONTINUOUS,
                            name='h_'+str(headCounter))
                else:
                    negVar = self.model.addVar(obj=0, lb = 0.0, ub = 1.0,
                            vtype=grb.GRB.BINARY,
                            name='h_'+str(headCounter))
                headCounter += 1

                self.headDict[RuleGrounding.pred_str(head)] = [headVar, negVar, [rvar]]
                self.headDict[RuleGrounding.pred_str(negation)] = [negVar, headVar, []]

                # BinaryPredicate case
                if (ruleGrounding.is_binary_head and 
                        (ruleGrounding.head['target_pos'] is None or 
                         ruleGrounding.head['ttype'] == LabelType.Binary)
                    ):

                    self.headDict[RuleGrounding.pred_str(negation)][2] = [rvar2]

                    if ruleGrounding.head['ttype'] == LabelType.Binary:
                        abstraction = copy.deepcopy(ruleGrounding.head)
                        abstraction['arguments'][abstraction['target_pos']] = "PH"
                        if (str(abstraction) not in self.binaryDict):
                            self.binaryDict[str(abstraction)] = \
                                [(headVar, ruleGrounding.head)]
                        else:
                            self.binaryDict[str(abstraction)].append((headVar,
                                ruleGrounding.head))
                    else:
                        if ruleGrounding.head['name'] not in self.binaryDict:
                            self.binaryDict[ruleGrounding.head['name']] = \
                                    [(headVar, ruleGrounding.head)]
                        else:
                            self.binaryDict[ruleGrounding.head['name']].append((headVar,
                                ruleGrounding.head))

                # Multi-Class/Multi-Label
                elif ruleGrounding.head['target_pos'] is not None \
                    and ruleGrounding.head['ttype'] == LabelType.Multiclass:
                    # generalize this predicate by replacing the concrete
                    # target arguement with "PH" (placeholder)
                    # TODO: better representation than "PH"
                    abstraction = copy.deepcopy(ruleGrounding.head)
                    # print abstraction
                    # TO-DO: make sure this target_pos is consistently correct
                    abstraction['arguments'][abstraction['target_pos']] = "PH"

                    if (str(abstraction) not in self.multiClassDict):
                        self.multiClassDict[str(abstraction)] = \
                            [(headVar, ruleGrounding.head)]
                    else:
                        self.multiClassDict[str(abstraction)].append((headVar,
                            ruleGrounding.head))
                elif ruleGrounding.head['target_pos'] is not None \
                        and ruleGrounding.head['ttype'] == LabelType.Multilabel:
                    # generalize this predicate by replacing the concrete
                    # target arguement with "PH" (placeholder)
                    # TODO: better representation than "PH"
                    abstraction = copy.deepcopy(ruleGrounding.head)
                    # print abstraction
                    # TO-DO: make sure this target_pos is consistently correct
                    abstraction['arguments'][abstraction['target_pos']] = "PH"

                    if (str(abstraction) not in self.multiLabelDict):
                        self.multiLabelDict[str(abstraction)] = \
                            [(headVar, ruleGrounding.head)]
                    else:
                        self.multiLabelDict[str(abstraction)].append((headVar,
                            ruleGrounding.head))
            else:
                self.headDict[RuleGrounding.pred_str(head)][2].append(rvar)

                # BinaryPredicate case
                if (ruleGrounding.is_binary_head and
                        (ruleGrounding.head['target_pos'] is None or
                         ruleGrounding.head['ttype'] == LabelType.Binary)
                    ):

                    self.headDict[RuleGrounding.pred_str(negation)][2].append(rvar2)

        self.model.update()

        self.ruleVarCounter = ruleVarCounter 
        debugout.write(str(self.headDict) + '\n')
        debugout.write(str(self.multiClassDict) + '\n')
        debugout.write(str(self.multiLabelDict) + '\n')
        debugout.write(str(self.binaryDict) + '\n')
        debugout.close()

        #print ("Added {} rule variables".format(ruleVarCounter))
        #print ("Added {} head variables".format(headCounter))

    def addRuleVariable(self):
        if self.relaxation != 'ILP':
            rvar = self.model.addVar(
                    obj=0,
                    lb = 0.0, ub = 1.0, vtype=grb.GRB.CONTINUOUS,
                    name='r_'+str(self.ruleVarCounter))
        else:
            rvar = self.model.addVar(
                    obj=0,
                    lb = 0.0, ub = 1.0, vtype=grb.GRB.BINARY,
                    name='r_'+str(self.ruleVarCounter))

        self.model.update()
        self.ruleVarCounter += 1
        return rvar

    def addHeadVariable(self, head, rvar):
        negation = copy.deepcopy(head)
        negation['isneg'] = not negation['isneg']

        headCounter = len(self.headDict)

        if self.relaxation == 'LP':
            headVar = self.model.addVar(obj=0, lb = 0.0, ub = 1.0,
                    vtype=grb.GRB.CONTINUOUS,
                    name='h_'+str(headCounter))
        else:
            headVar = self.model.addVar(obj=0, lb = 0.0, ub = 1.0,
                    vtype=grb.GRB.BINARY,
                    name='h_'+str(headCounter))
        headCounter += 1
        if self.relaxation == 'LP':
            negVar = self.model.addVar(obj=0, lb = 0.0, ub = 1.0,
                    vtype=grb.GRB.CONTINUOUS,
                    name='h_'+str(headCounter))
        else:
            negVar = self.model.addVar(obj=0, lb = 0.0, ub = 1.0,
                    vtype=grb.GRB.BINARY,
                    name='h_'+str(headCounter))
        headCounter += 1

        self.headDict[RuleGrounding.pred_str(head)] = [headVar, negVar, [rvar]]
        self.headDict[RuleGrounding.pred_str(negation)] = [negVar, headVar, []]
        self.model.update()

    def addConstraints(self):
        """
        Adds constraints based on all rule groundings in this instance.
        This method should be applied after addVariables().
        TO-DO: modularize constraints for readability
        """
        #print("Adding Constraints...")
        constrCounter = 0

        # hard constraints
        multiclass_hard_constraints = {}

        n_seeds = 0
        seed_set = set([])

        # Add constraint rules

        #print "heads before constraints", len(self.headDict)

        constraintHeads = set([])
        for constrGrounding in self.constraintGroundings:
            unobserved = [pred for pred in constrGrounding.body if not pred['obs']]
            if len(unobserved) == 0 and RuleGrounding.pred_str(constrGrounding.head) not in self.headDict:
                rvar = self.addRuleVariable()
                self.addHeadVariable(constrGrounding.head, rvar)
                constraintHeads.add(RuleGrounding.pred_str(constrGrounding.head))
        #print constraintHeads
        #exit()
        #print "heads after constraints", len(self.headDict)
        #exit()
        same_constaints = 0
        for constrGrounding in self.constraintGroundings:
            body_variables = []

            unobserved = [pred for pred in constrGrounding.body if not pred['obs']]
            inheads = [pred for pred in unobserved if RuleGrounding.pred_str(pred) in self.headDict]

            if len(unobserved) != len(inheads):
                continue

            body_variables = [self.headDict[RuleGrounding.pred_str(pred)][0] for pred in unobserved]
            hvar = self.headDict[RuleGrounding.pred_str(constrGrounding.head)][0]

            if len(body_variables) > 0:
                self.model.addConstr(grb.quicksum(body_variables), grb.GRB.LESS_EQUAL,
                                     len(body_variables)-1 + hvar, "c_"+str(constrCounter))
                constrCounter += 1
                same_constaints += 1
                #print constrGrounding
            elif constrGrounding.head['ttype'] == 1:
                # keep track of multiclass elements in head of rule
                # TO-DO: keep track of multilabel also
                non_targets = [constrGrounding.head['arguments'][i]
                                for i in range(len(constrGrounding.head['arguments']))
                                if i != constrGrounding.head['target_pos']]
                key = constrGrounding.head['name'] + str(non_targets)
                if key not in multiclass_hard_constraints:
                    multiclass_hard_constraints[key] = []
                multiclass_hard_constraints[key].append(self.headDict[RuleGrounding.pred_str(constrGrounding.head)][0])
                # constraint will be added later
            else:
                # enforcing the head
                n_seeds += 1
                self.model.addConstr(hvar, grb.GRB.EQUAL, 1, "c_"+str(constrCounter))
                constrCounter += 1
                seed_set.add(hvar)

        #print "SEEDED", n_seeds
        #print "AGR/DGR", same_constaints
        #print len(constraintHeads)
        #exit()
        # add multiclass hard constraints
        # TO-DO: add multilabel
        for key in multiclass_hard_constraints:
            self.model.addConstr(grb.quicksum(multiclass_hard_constraints[key]),
                                 grb.GRB.EQUAL, 1, "c_"+str(constrCounter))
            constrCounter += 1
            #print "Added multiclass constr for", key
        #print "Multiclass hard constraints", len(multiclass_hard_constraints)


        # imply constr
        ruleVarCounter = 0

        #print len(self.ruleGroundings.keys())
        seeded_B = 0
        deactivated_B = 0
        for ruleGrounding in self.ruleGroundings.keys():

            r = self.model.getVarByName('r_'+str(ruleVarCounter))

            unobs = [pred for pred in ruleGrounding.body if not pred['obs']]
            found = [pred for pred in unobs if RuleGrounding.pred_str(pred) in self.headDict]

            if len(unobs) != len(found):
                self.model.addConstr(r, grb.GRB.EQUAL, 0, "c_"+str(constrCounter))
                constrCounter += 1
                deactivated_B += len(unobs)
            else:
                for pred in found:
                    if RuleGrounding.pred_str(pred) in constraintHeads:
                        seeded_B += 1
                    h = self.headDict[RuleGrounding.pred_str(pred)][0]
                    self.model.addConstr(h, grb.GRB.GREATER_EQUAL,
                                         r, "c_"+str(constrCounter))
                    constrCounter += 1

            ruleVarCounter += 1

            # do not need to have imply constr for negation of binary head
            if (ruleGrounding.is_binary_head and
                    (ruleGrounding.head['target_pos'] is None or
                     ruleGrounding.head['ttype'] == LabelType.Binary)
                ):
                #print "HERE"
                ruleVarCounter += 1

        #print "seeded_B", seeded_B
        #print "deactivated_B"

        # negation constr, rule/head constr
        neg_constr = 0; impl_constraints = 0
        for head in self.headDict.keys():
            # h
            headVar = self.headDict[head][0]
            if (head.startswith("~")):
                # neg
                negVar = self.headDict[head][1]
                self.model.addConstr(headVar + negVar, grb.GRB.EQUAL,
                                     1, "c_"+str(constrCounter))
                constrCounter += 1
                neg_constr += 1
            # [r1, r2, ...]
            rs = self.headDict[head][2]
            if (len(rs) == 0):
                continue
            # r1 + r2 + ...
            rsum = grb.quicksum(rs)
            for r in rs:
                # h >= r_i
                self.model.addConstr(headVar, grb.GRB.GREATER_EQUAL,
                                     r, "c_"+str(constrCounter))
                constrCounter += 1
            # h <= sum(r_i)
            self.model.addConstr(headVar, grb.GRB.LESS_EQUAL,
                                 rsum, "c_"+str(constrCounter))
            constrCounter += 1

        #print "Negation constr", neg_constr

        n_multiclass = 0; n_multilabel = 0
        # Binary/Multi-Class/Multi-Label constr
        for abstra in self.multiClassDict.keys():
            vs = [item[0] for item in self.multiClassDict[abstra]]
            #print abstra, len(vs)
            vsum = grb.quicksum(vs)
            self.model.addConstr(vsum, grb.GRB.EQUAL, 1,
                                 "c_"+str(constrCounter))
            constrCounter += 1
            n_multiclass += 1

        for abstra in self.multiLabelDict.keys():
            vs = [item[0] for item in self.multiLabelDict[abstra]]
            vsum = grb.quicksum(vs)
            self.model.addConstr(vsum, grb.GRB.GREATER_EQUAL, 1,
                                 "c_"+str(constrCounter))
            constrCounter += 1
            n_multilabel += 1
        #print "multiclass constraints", n_multiclass
        #print "multilabel constraints", n_multilabel

        #print "NUMBER OF VARIABLES", len(self.headDict)
        self.model.update()
        # print("Added " + str(constrCounter) + " constraints")


    def encode(self):
        self.createModel("testModel")
        self.addVariables()
        self.addConstraints()

    def optimize(self):
        """
        Optimizes the current model.
        """
        self.model.optimize()

    def get_predictions(self):
        predictions = set([])
        for head in self.headDict.keys():
            h = self.headDict[head][0]

            try:
                pred_value = h.getAttr('x')
                if round(pred_value) == 1 and not head.startswith('~'):
                    predictions.add(head)
            except:
                continue
        return predictions

    def get_binary_metrics(self, dictionary, gold_heads_dict):
        gold_heads = [RuleGrounding.pred_str(head) for head in gold_heads_dict]
        gold_heads = set(gold_heads)

        metrics = {}; predictions = set([])
        for (key, value_ls) in dictionary:
            for tup in value_ls:
                var, head = tup[0], tup[1]

                if head['name'] not in metrics:
                    metrics[head['name']] = {'gold_data': [], 'pred_data': []}

                head_str = RuleGrounding.pred_str(head)

                # Gold label
                if head_str in gold_heads:
                    metrics[head['name']]['gold_data'].append(1)
                else:
                    metrics[head['name']]['gold_data'].append(0)

                # Pred label
                pred_value = var.getAttr('x')
                metrics[head['name']]['pred_data'].append(int(pred_value))

                if round(pred_value) == 1:
                    predictions.add(head_str)

        return predictions, metrics

    def get_multi_metrics(self, dictionary, gold_heads, binary_predicates):
        metrics = {}
        predictions = set([])

        # load dictionary of gold predictions
        gold_data = {}

        for elem in gold_heads:
            classif_name = elem['name']
            # skip binary predicates
            if classif_name in binary_predicates:
                continue

            if classif_name not in gold_data:
                gold_data[classif_name] = {}
            id_ = ",".join(str(elem['arguments'][i]) for i in range(0, len(elem['arguments'])) if i != elem['target_pos'])
            #print elem
	    label = elem['arguments'][elem['target_pos']]
            gold_data[classif_name][id_] = label

        # load dictionary of pred labels
        pred_data = {}

        for (key, value_ls) in dictionary:
            for tup in value_ls:
                var, head = tup[0], tup[1]

                # skip things we are not evaluating on
                if head['name'] not in gold_data:
                    continue

                if head['name'] not in metrics:
                    metrics[head['name']] = {'gold_data': [], 'pred_data': []}
                    pred_data[head['name']] = {}

                k_class = head['arguments'][head['target_pos']]
                curr_id = ",".join(str(head['arguments'][i]) for i in range(0, len(head['arguments'])) if i != head['target_pos'])

                pred_value = var.getAttr('x')
                if round(pred_value) == 1:
                    pred_data[head['name']][curr_id] = k_class
                    predictions.add(RuleGrounding.pred_str(head))
        '''
        print "--- Pred"
        for h in predictions:
            print h
        '''

        # create ordered list of predictions to be used in sklearn
        for classif_name in gold_data:

            if classif_name not in pred_data or len(pred_data[classif_name]) != len(gold_data[classif_name]):
                continue

            for id_ in gold_data[classif_name]:
                gold_label = gold_data[classif_name][id_]
                pred_label = pred_data[classif_name][id_]
                metrics[classif_name]['gold_data'].append(gold_label)
                metrics[classif_name]['pred_data'].append(pred_label)

        return predictions, metrics


    # TO-DO: re-add and test binary case
    def evaluate(self, gold_heads, binary_predicates=[]):
        '''
        self.model.computeISS()
        print('The following constraints cannot be satisfied:')
        for c in m.getConstrs():
            if c.ISSConstr:
                print('%s' % c.constrName)
        '''

        # Evaluate multiclass classifiers
        metrics = {}
        _, metrics_multiclass = self.get_multi_metrics(self.multiClassDict.items(),
                gold_heads, binary_predicates)
        _, metrics_binary = self.get_binary_metrics(self.binaryDict.items(), gold_heads)

        metrics.update(metrics_multiclass)
        metrics.update(metrics_binary)

        return metrics
