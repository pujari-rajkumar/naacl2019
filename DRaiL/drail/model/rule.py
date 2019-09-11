"""
This module defines the representation of a rule template and a \
rule grounding
"""
class RuleTemplate(object):


    def __init__(self, body_predicates, head_predicate,
                filters, feature_functions, lambda_, target,
                isconstr, split_on, label_type, dbfunc,
                scoring_function):
    # def __init__(self, body_predicates, head_predicate, target_predicates,
    # neural_net, feature_file, feature_class, feature_functions, constr):

        """
        Create a rule template and its attributes. A rule consist of an
        implicative clause of the form: p_1 and p_2 and ...  p_n implies h.
        Where p_i and h are predicates

        Args:
            body_predicates: predicates p1, .., pn
            head_predicate: predicate h
            target_predicates: target predicates for prediction
            neural_net: neural network to be used for
                        predicting target predicates
            constr: whether this rule template defines constraints
        """
        self.body = body_predicates
        self.head = head_predicate
        #self.targets = target_predicates

        # keep a dict indexable by predicate name
        self.predicate_index = {}

        # set of variables in the rule
        self.variable_set = set([])
        for pred in (self.body + [self.head]):
            if pred.name in self.predicate_index:
                self.predicate_index[pred.name].append(pred)
            else:
                self.predicate_index[pred.name] = [pred]
            self.variable_set |= pred.var_set

        # filter predicates for sampling
        self.filters = filters

        # rule weight
        self.lambda_ = lambda_

        # target variable
        self.target = target

        # self.neural_net = neural_net
        self.isconstr = isconstr
        self.split_on = split_on
        self.split_on_ttype = label_type
        self.split_on_classes = []

        self.feat_functions = feature_functions
        self.feat_vector_sz = None
        # temporary decision
        #self.head_pred_type = head_pred_type

        # database configuration to be able to hcode
        self.dbfunc = dbfunc

        self.scoring_function = scoring_function

    # does it matter if targets are equal?
    # for now i think not
    def __eq__(self, other):
        """
        Check if `other` rule template is equal. Rule templates are equal if
        they have the same body and head

        Args:
            other: rule template to compare

        Returns:
            True if they are equal, False otherwise
        """
        if len(self.body) != len(other.body) or \
           self.head != other.head:
               return False
        for p, p_prime in zip(self.body, other.body):
            if p != p_prime:
                return False
        return True

    def __neq__(self, other):
        """
        Check if `other` rule template is not equal

        Args:
            other: rule template to compare

        Returns:
            True if they are not equal, False otherwise
        """
        return not self.__eq__(other)

    def __str__(self):
        """
        Obtain a string representation of a rule template of the form:
        p1 & p2 & p3 => h

        Returns:
            A string representation of the template
        """
        if len(self.body) > 0:
            ret = str(self.body[0])
            for i in range(1, len(self.body)):
                ret += " & " + str(self.body[i])
            ret += " => " + str(self.head)
            return ret
        else:
            return ""

    def __repr__(self):
        """
        Use the string representation for rule templates when
        printing collections of rules

        Returns:
            A string representation of the template
        """
        return str(self)

    def target_vars(self, include_head=True):
        tvars = []
        if include_head:
            predicates = self.body + [self.head]
        else:
            predicates = self.body
        for pred in predicates:
            tv = pred.label_var()
            if tv is not None:
                tvars.append(tv.arg)
        return tvars

    def target_constants(self, include_head=True):
        tconst = []
        if include_head:
            predicates = self.body + [self.head]
        else:
            predicates = self.body
        for pred in predicates:
            tc = pred.label_const()
            if tc is not None:
                tconst.append(tc)
        return tconst

    def target_preds(self, include_head=True):
        tpreds = []
        if include_head:
            predicates = self.body + [self.head]
        else:
            predicates = self.body
        for pred in predicates:
            if not pred.isobs:
                tpreds.append(pred.name)
        return tpreds

    def body_cardinality(self):
        cardinality = []
        for pred in self.body:
            for var in pred.variables:
                if not var.isobs:
                    cardinality.append(int(var.label.n_classes))
            if not pred.isobs:
                cardinality.append(2)
        if len(cardinality) == 0:
            return 1
        else:
            return reduce(lambda x,y: x * y, cardinality)


    def head_cardinality(self):
        cardinality = []
        for var in self.head.variables:
            if not var.isobs:
                cardinality.append(int(var.label.n_classes))
        if not self.head.isobs:
            cardinality.append(2)

        if len(cardinality) == 0:
            return 1
        else:
            return reduce(lambda x,y: x * y, cardinality)


class RuleGrounding(object):

    def __init__(self, body_groundings, head_grounding, uid_template, is_binary_head,
                 target, isgroundtruth=False):
        """
        Create a rule grounding and its attributes. A rule consist of an
        implicative clause of the form: p_1 and p_2 and ...  p_n implies h.
        Where p_i and h are predicate groundings.

        Args:
            uid_grounding: unique identifier for the grounding
            uid_template: unique identifier for the template
            uid_instance: unique identifier for the inference instance
            body_groundings: left hand side predicate groundings p1, .., pn
            head_grounding: right hand side predicate grounding h
            isgroundtruth: 1 if template is observed in the data, 0 otherwise
        """
        self.uid_template = uid_template
        self.body = body_groundings
        self.head = head_grounding
        self.isgroundtruth = isgroundtruth
        self.is_binary_head = is_binary_head
        self.feature_repr = None
        self.target = target

    def build_predicate_index(self):
        """
        Build an index of predicate names and groundings to be used for
        feature extraction. More than one predicate grounding might have
        the same name, so lists are kept.
        """
        self.predicate_index = {}
        for pred_grounding in self.body:
            if pred_grounding['name'] not in self.predicate_index:
                self.predicate_index[pred_grounding['name']] = []
            # print "debug", pred_grounding
            self.predicate_index[pred_grounding['name']].append(pred_grounding)
        # print "***"
            # print pred_grounding
        if self.head is not None:
            self.predicate_index[self.head['name']] = []
            self.predicate_index[self.head['name']].append(self.head)

        # print self.predicate_index


    def build_body_predicate_dic(self):
        """
        Build an index of predicate names and groundings to be used for
        feature extraction. More than one predicate grounding might have
        the same name, so lists are kept.
        """
        self.body_predicate_dic = {}
        for pred_grounding in self.body:
            if pred_grounding['name'] not in self.body_predicate_dic:
                self.body_predicate_dic[pred_grounding['name']] = []
            self.body_predicate_dic[pred_grounding['name']].append(pred_grounding)

        '''
        if self.head is not None:
            self.predicate_index[self.head['name']] = []
            self.predicate_index[self.head['name']].append(self.head)
        '''
        # print "Na"
        # print self.body_predicate_dic

    def has_predicate(self, name):
        """
        Check if a rule grounding contains a given predicate

        Args:
            name: predicate name

        Return:
            True if a predicate with that name exists in the
                 rule grounding's body,
            False otherwise
        """
        return name in self.predicate_index

    def has_body_predicate(self, name):
        """
        Check if a rule grounding's body contains a given predicate

        Args:
            name: predicate name

        Return:
            True if a predicate with that name exists in the rule
                 grouding's body,
            False otherwise
        """
        return name in self.body_predicate_dic

    def get_predicates(self, name):
        """
        Get the list of predicate groundings by name

        Args:
            name: predicate name

        Returns:
            a list of predicate groundings with the given name
        """
        return self.predicate_index[name]

    def get_body_predicates(self, name):
        """
        Get the list of body predicate groundings

        Returns:
            a list of predicate groundings with the given name
        """
        # print self.body_predicate_dic
        if len(self.body_predicate_dic) == 0:
            self.body_predicate_dic()
        return self.body_predicate_dic[name]

    def get_head_predicate(self):
        """
        Get the head predicate grounding

        Returns:
            a single element of predicate grounding
        """
        # return self.head.arguments
        return self.head

    def set_feature_repr(feature_repr):
        self.feature_repr = feature_repr

    def get_feature_repr():
        return self.feature_repr

    def __eq__(self, other):
        """
        Check if `other` rule grounding is equal. Rule groundings are equal if
        they have the same body and head

        Args:
            other: rule grounding to compare

        Returns:
            True if they are equal, False otherwise
        """
        if len(self.body) != len(other.body) or \
           self.head != other.head:
               return False
        for p, p_prime in zip(self.body, other.body):
            if p != p_prime:
                return False
        return True

    def __neq__(self, other):
        """
        Check if `other` rule grounding is not equal

        Args:
            other: rule grounding to compare

        Returns:
            True if they are not equal, False otherwise
        """
        return not self.__eq__(other)

    @staticmethod
    def pred_str(dict_pred, trim=False):
        temp = []
        for x in dict_pred['arguments']:
            if isinstance(x, unicode):
                x_str = x.encode('utf-8')
            else:
                x_str = str(x)
            if len(x_str) > 10 and trim:
                x_str = x_str[:10] + "..." + x_str[len(x_str)/2] + "..." + x_str[-11:-1]
            temp.append(x_str)
        temp = ",".join(temp)
        if dict_pred['isneg']:
            ret = "~{0}({1})"
        else:
            ret = "{0}({1})"
        if not dict_pred['obs']:
            ## body groundings don't have this info ?
            if 'target_type' in dict_pred:
                if dict_pred['target_type'] == LabelType.Multiclass:
                    ret += "*" + "Multiclass" + "-" + str(dict_pred['target_pos'])
                elif dict_pred['target_type'] == LabelType.Multilabel:
                    ret += "*" + "Multilabel" + "-" + str(dict_pred['target_pos'])
        return ret.format(dict_pred['name'], temp)

    def __str__(self):
        """
        Obtain a string representation of a rule grounding of the form:
        p1 & p2 & p3 => h

        Returns:
            A string representation of the rule
        """
        if len(self.body) > 0:
            ret = RuleGrounding.pred_str(self.body[0], trim=True)
            for i in range(1, len(self.body)):
                ret += " & " + RuleGrounding.pred_str(self.body[i], trim=True)
            ## body groundings won't have a head
            if self.head is not None:
                ret += " => " + RuleGrounding.pred_str(self.head, trim=True)
            if self.isgroundtruth:
                ret += " ***"
            return ret
        else:
            return ""

    def _str_repr(self):
        """
        Obtain a string representation of a rule grounding of the form:
        p1 & p2 & p3 => h

        Returns:
            A string representation of the rule
        """
        if len(self.body) > 0:
            ret = RuleGrounding.pred_str(self.body[0])
            for i in range(1, len(self.body)):
                ret += " & " + RuleGrounding.pred_str(self.body[i], trim=True)
            ## body groundings won't have a head
            if self.head is not None:
                ret += " => " + RuleGrounding.pred_str(self.head, trim=True)
            if self.isgroundtruth:
                ret += " ***"
            return ret
        else:
            return ""

    def __repr__(self):
        """
        Use the string representation for rule grounding when
        printing collections of rules
            A string representation of the rule
        """
        return _str_repr(self)

    def to_json(self):
        ret = {}
        ret["uid_template"] = self.uid_template
        ret["body"] = self.body_groundings
        ret["head"] = self.head_groundings
        ret["isgroundtruth"] = self.isgroundtruth
        ret["is_binary_head"] = self.is_binary_head
        ret["feature_repr"] = self.feature_repr
        ret["target"] = self.target
        ret["predicate_index"] = self.predicate_index
        ret["body_predicate_dic"] = self.body_predicate_dic
        return ret

def rg_from_json(jsonfile):
    with open(jsonfile) as data_file:
        data = json.load(data_file)
    rg = RuleGrounding(data['body'], data['head'], data['uid_template'],
                       data['is_binary_head'], data['target'],
                       isgroundtruth=data['isgroundtruth'])
    rg.feature_repr = data['feature_repr']
    rg.predicate_index = data['predicate_index']
    rg.body_predicate_dic = data['body_predicate_dic']
    return rg
