import json
import ply.yacc as yacc
import copy
from lexer import Lexer

# load classes from model
from ..model.argument import ArgumentDefinition, ArgumentType, Argument
from ..model.label import Label, LabelType
from ..model.predicate import PredicateTemplate
from ..model.rule import RuleTemplate
from ..model.equation import EquationTemplate, EquationElement
from ..model.feature_function import FeatureFunction, FeatureType
from ..model.scoring_function import ScoringFunction, ScoringType
#from ..neuro.nn_model import NeuralNetworks

class Parser(object):

    def __init__(self):
        self.lexer = Lexer()
        self.tokens = self.lexer.tokens
        self.literals = self.lexer.literals
        self.entity_arguments = {}
        self.predicate_arguments = {}
        self.predicate_entities = {}

        self.label_classes = {}
        self.label_types = {}
        self.files = {}
        self.rulesets = []
        self.groupby = []

        self.dbmodule = None
        self.dbclass = None
        self.femodule = None
        self.feclass = None
        self.scmodule = None

    def p_model(self, p):
        "model : instructionlist"
        pass

    def p_instructionlist(self, p):
        '''
        instructionlist : instructionlist instruction ';'
                        | instruction ';'
        '''
        pass

    def p_instruction_entity(self, p):
        "instruction : ENTITY ':' VARSTRING ',' ARGUMENTS ':' '[' argumentdeflist ']'"
        name = p[3][1:len(p[3]) - 1]
        self.entity_arguments[name] = p[8]

    ## rules for parsing "predicate" creation
    def p_instruction_predicate(self, p):
        "instruction : PREDICATE ':' VARSTRING ',' ARGUMENTS ':' '[' VARPRED ',' VARPRED ']'"
        name = p[3][1:len(p[3]) - 1]

        arg1 = copy.deepcopy(self.entity_arguments[p[8]][0])
        arg2 = copy.deepcopy(self.entity_arguments[p[10]][0])
        arg1.name = arg1.name + "_1"
        arg2.name = arg2.name + "_2"
        self.predicate_arguments[name] = [arg1, arg2]
        self.predicate_entities[name] = [p[8], p[10]]

    def p_argument_def_list(self, p):
        '''
        argumentdeflist : argumentdeflist ',' argumentdef
                        | argumentdef
        '''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1]
            p[0].append(p[3])

    def p_argument(self, p):
        "argumentdef : VARSTRING ':' ':' type"
        arg_name = p[1][1:len(p[1]) - 1]
        p[0] = ArgumentDefinition(arg_name, p[4])

    def p_type_string(self, p):
        "type : ARGSTRING"
        p[0] = ArgumentType.String

    def p_type_string_id(self, p):
        "type : ARGSTRINGID"
        p[0] = ArgumentType.UniqueString

    def p_type_int(self, p):
        "type : ARGINT"
        p[0] = ArgumentType.Integer

    def p_type_double(self, p):
        "type : ARGDOUBLE"
        p[0] = ArgumentType.Double

    def p_type_id(self, p):
        "type : ARGID"
        p[0] = ArgumentType.UniqueID

    ## rules for parsing "label" creation
    def p_instruction_label(self, p):
        "instruction : LABEL ':' VARSTRING ',' LBCLASSES ':' NUMBER ',' LBTYPE ':' labeltype"
        label = p[3][1:len(p[3]) - 1]
        self.label_classes[label] = p[7]
        self.label_types[label] = p[11]

    def p_labeltype_multilabel(self, p):
        "labeltype : LBMULTILABEL"
        p[0] = LabelType.Multilabel

    def p_labeltype_multiclass(self, p):
        "labeltype : LBMULTICLASS"
        p[0] = LabelType.Multiclass

    def p_labeltype_binary(self, p):
        "labeltype : LBBINARY"
        p[0] = LabelType.Binary

    ## rules for parsing "load" instructions

    def p_instruction_load(self, p):
        "instruction : LOAD ':' VARSTRING ',' FILE ':' VARSTRING"
        name = p[3][1:len(p[3]) - 1]
        filename = p[7][1:len(p[7]) - 1]
        self.files[name] = filename

    ## rules for parsing "rules" over predicates

    def p_instruction_ruleset(self, p):
        "instruction : RULESET '{' rulelist '}' GROUPBY ':' GROUPARG"
        self.rulesets.append(p[3])
        dot_index = p[7].index('.')
        gb_table = p[7][:dot_index]
        gb_arg = p[7][dot_index + 1:]
        self.groupby.append((gb_table, gb_arg))

    def p_instruction_dbmodule(self, p):
        "instruction : DBMODULE ':' VARSTRING"
        self.dbmodule = p[3][1:len(p[3]) - 1]

    def p_instruction_dbclass(self, p):
        "instruction : DBCLASS ':' VARSTRING"
        self.dbclass = p[3][1:len(p[3]) - 1]

    def p_instruction_femodule(self, p):
        "instruction : FEMODULE ':' VARSTRING"
        self.femodule = p[3][1:len(p[3]) - 1]

    def p_instruction_feclass(self, p):
        "instruction : FECLASS ':' VARSTRING"
        self.feclass = p[3][1:len(p[3]) - 1]

    def p_instruction_scmodule(self, p):
        "instruction : SCMODULE ':' VARSTRING"
        self.scmodule = p[3][1:len(p[3]) - 1]

    def p_rulelist(self, p):
        '''
        rulelist : rulelist rule ';'
                 | rule ';'
        '''
        if len(p) == 3:
            p[0] = {}
            if p[1].isconstr:
                p[0]['constr'] = [p[1]]
            else:
                p[0]['rule'] = [p[1]]
        else:
            p[0] = p[1]
            if (type(p[2]) is RuleTemplate and p[2].isconstr) or (type(p[2]) is EquationTemplate):
                if 'constr' not in p[0]:
                    p[0]['constr'] = [p[2]]
                else:
                    p[0]['constr'].append(p[2])
            else:
                p[0]['rule'].append(p[2])

    def p_rule(self, p):
        '''
        rule : RULE ':' proposition filters lambda scoring featureconf dbconf splitconf target
        '''
        (body, head) = p[3]
        filters = p[4]; lambda_ = p[5]; scoring = p[6]
        feature_funcs = p[7]; dbfunc = p[8]
        (split_on, label_type) = p[9]; target = p[10]
        p[0] = RuleTemplate(body, head, filters, feature_funcs,
                            lambda_, target, isconstr=False,
                            split_on=split_on, label_type=label_type,
                            dbfunc=dbfunc, scoring_function=scoring)

    def p_lambda(self, p):
        '''
        lambda : LAMBDA ':' FLOAT ','
               |
        '''
        if len(p) == 5:
            p[0] = float(p[3])
        else:
            p[0] = 1.0

    def p_splitconf(self, p):
        '''
        splitconf : SPLITCLASSIF ':' VARINSTANCE ':' ':' labeltype ','
                  |
        '''
        if len(p) == 8:
            p[0] = (p[3], p[6])
        else:
            p[0] = (None, None)

    def p_target(self, p):
        '''
        target : TARGET ':' VARINSTANCE
        '''
        p[0] = p[3]

    def p_filters(self, p):
        '''
        filters : CONDS ':' '[' condlist ']' ','
                |
        '''
        if len(p) == 1:
            p[0] = []
        else:
            p[0] = p[4]

    def p_cond_list(self, p):
        '''
        condlist : condlist ',' condition
                 | condition
        '''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1]
            p[0].append(p[3])

    def p_condition_string(self, p):
        "condition : VARINSTANCE '.' VARARG '=' VARSTRING"
        p[0] = (p[1], p[3], "'" + p[5][1:len(p[5]-1)] + "'")

    def p_condition_number(self, p):
        "condition : VARINSTANCE '.' VARARG '=' NUMBER"
        p[0] = (p[1], p[3], p[5])

    def p_condition_float(self, p):
        "condition : VARINSTANCE '.' VARARG '=' FLOAT"
        p[0] = (p[1], p[3], p[5])

    def p_features(self, p):
        "featureconf : FEFUNC ':' '[' featurelist ']' ','"
        p[0] = p[4]

    def p_dbfunction(self, p):
        '''
        dbconf : DBFUNC ':' VARSTRING ','
               | DBFUNC ':' VARSTRING
               |
        '''
        if len(p) == 1:
            p[0] = None
        else:
            p[0] = p[3][1:len(p[3]) - 1]

    def p_scoring_function(self, p):
        '''
        scoring : NETWORK ':' VARSTRING ','
                | SCORE ':' RULESET '.' NUMBER ','
                | SCORE ':' VARSTRING ','
        '''
        if len (p) == 5:
            if p[1] == "network":
                p[0] = ScoringFunction(ScoringType.NNet)
            elif p[1] == "score":
                p[0] = ScoringFunction(ScoringType.Func,
                                       scfunc=p[3][1:len(p[3])-1])
        elif len(p) == 7:
            p[0] = ScoringFunction(ScoringType.NNetRef, scref=int(p[5]))

    def p_feature_list(self, p):
        '''
        featurelist : featurelist ',' featurefunc
                    | featurefunc
        '''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1]
            p[0].append(p[3])


    def p_feature_func(self, p):
        '''
        featurefunc : EMBEDDING '(' VARSTRING ',' VARSTRING ',' VARSTRING ',' VARSTRING ',' VARSTRING ')'
                    | VECTOR '(' VARSTRING ')'
                    | INPUT '(' VARSTRING ')'
        '''
        feature_name = p[3][1:len(p[3])-1]
        if len(p) == 13:
            emb_container = p[5][1:len(p[5])-1]
            vocab_size = p[7][1:len(p[7])-1]
            emb_size = p[9][1:len(p[9])-1]
            vocab_index = p[11][1:len(p[11])-1]
            p[0] = FeatureFunction(feature_name, FeatureType.Embedding,
                                   emb_container, vocab_size, emb_size, vocab_index)
        elif p[1] == "vector":
            p[0] = FeatureFunction(feature_name, FeatureType.Vector)
        elif p[1] == "input":
            p[0] = FeatureFunction(feature_name, FeatureType.Input)

    def p_hard_constraint_rule(self, p):
        '''
        rule : HARDCONSTRAINT ':' proposition filters dbconf
        '''
        (body, head) = p[3]
        print p[5]
        if len(p) == 8:
            p[0] = RuleTemplate(body, head, p[5], None, float('inf'), None, True, None, None, p[7], None)
        else:
            p[0] = RuleTemplate(body, head, [], None, float('inf'), None, True, None, None, p[5], None)


    # OJO: this would change when head hand side stops being a single predicate
    def p_proposition(self, p):
        '''
        proposition : conjunction IMPL instance ','
        proposition : conjunction IMPL instance
        '''
        # returns (body_list, head_instance)
        p[0] = (p[1], p[3])

    def p_conjunction(self, p):
        '''
        conjunction : conjunction '&' instance
                    | instance
        '''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1]
            p[0].append(p[3])

    def p_instance(self, p):
        "instance : VARPRED '(' argumentlist ')'"
        p[0] = PredicateTemplate(p[1], p[3])

    def p_instance_neg(self, p):
        "instance : '~' VARPRED '(' argumentlist ')'"
        p[0] = PredicateTemplate(p[2], p[4], isneg=True)

    def p_instance_target(self, p):
        "instance : VARPRED '(' argumentlist ')' '^' '?'"
        p[0] = PredicateTemplate(p[1], p[3], isobs=False)

    def p_instance_neg_target(self, p):
        "instance : '~' VARPRED '(' argumentlist ')' '^' '?'"
        p[0] = PredicateTemplate(p[2], p[4], isneg=True, isobs=False)

    def p_instance_select(self, p):
        "instance : VARPRED '(' argumentlist ')' '*'"
        p[0] = PredicateTemplate(p[1], p[3], select=True)

    def p_instance_neg_select(self, p):
        "instance : '~' VARPRED '(' argumentlist ')' '*'"
        p[0] = PredicateTemplate(p[1], p[3], isneg=True, select=True)

    def p_instance_target_select(self, p):
        "instance : VARPRED '(' argumentlist ')' '^' '?' '*'"
        p[0] = PredicateTemplate(p[1], p[3], isobs=False, select=True)

    def p_instance_neg_target_select(self, p):
        "instance : '~' VARPRED '(' argumentlist ')' '^' '?' '*'"
        p[0] = PredicateTemplate(p[1], p[3], isneg=True, isobs=False, select=True)

    def p_argument_list(self, p):
        '''
        argumentlist : argumentlist ',' argument
                     | argument
        '''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1]
            p[0].append(p[3])

    def p_argument_variable(self, p):
        "argument : VARINSTANCE"
        p[0] = Argument(p[1], isconstant=False)

    def p_argument_constant(self, p):
        "argument : VARSTRING"
        p[0] = Argument(p[1][1:len(p[1]) - 1], isconstant=True)

    def p_argument_target(self, p):
        "argument : VARINSTANCE '^' VARPRED '?'"
        p[0] = Argument(p[1], isconstant=False, isobs=False,
                label=Label(p[3], self.label_types[p[3]], self.label_classes[p[3]]))

    def p_argument_constant_target(self, p):
        "argument : VARSTRING '^' VARPRED '?'"
        p[0] = Argument(p[1][1:len(p[1]) - 1], isconstant=True, isobs=False,
                label=Label(p[3], self.label_types[p[3]], self.label_classes[p[3]]))

    def p_error(self, p):
        if p is not None:
            print "Syntax error at '%s'" % p.value

    def build(self, **kwargs):
        self.lexer.build()
        self.parser = yacc.yacc(module=self, **kwargs)

    def parse(self, filename):
        with open(filename) as f:
            text = f.read()
            self.parser.parse(text)
