import sys
import os
import numpy as np
from drail.database import Database
from drail.parser import parser

def main():
    if len(sys.argv) < 4:
        sys.exit("Usage:" + " python " + sys.argv[0] +" [rule file] [data path] [filter_file] [-arg1 arg, -arg2 arg, ...]")

    filename = sys.argv[1]
    text = open(filename).read()
    par = parser.Parser()
    par.build()
    par.parse(filename)

    filter_ids = []
    with open(sys.argv[3]) as f:
        for line in f:
            filter_ids.append(line.strip())

    kwargs = {}
    for i in range(4, len(sys.argv), 2):
        kwargs[sys.argv[i][1:]] = sys.argv[i+1]

    db = Database()
    path = sys.argv[2]

    db.load_predicates(par.predicate_arguments, path, par.files)
    db.load_labels(par.label_types.keys(), path, par.files)

    for ruleset, group_by in zip(par.rulesets, par.groupby):
        for i, rule_template in enumerate(ruleset):

            X = []; Y = []

            if rule_template.isconstr:
                continue

            mod = __import__(rule_template.feat_file,fromlist=[rule_template.feat_class])
            fe_class = getattr(mod, rule_template.feat_class)

            if rule_template.head.isobs:
                # multiclass
                train_groundings = db.unfold_train_groundings(rule_template, filter_by=(group_by, filter_ids))
            else:
                # binary
                neg_train_groundings = db.unfold_train_groundings(rule_template, filter_by=(group_by, filter_ids), neg_head=True)
                pos_train_groundigs = db.unfold_train_groundings(rule_template, filter_by=(group_by, filter_ids), neg_gead=False)
                train_groundings = neg_train_groundings + pos_train_groundigs

            fe = fe_class(train_groundings, **kwargs)
            fe.build()

            grd_x = fe.extract(train_groundings[0], rule_template.feat_functions)
            rule_template.feat_vector_sz = len(grd_x)

            for instance_grd in train_groundings:
                grd_x = fe.extract(instance_grd, rule_template.feat_functions)
                X.append(grd_x)

                # when multiclass I need a label for the output
                if rule_template.head.isobs:
                    grd_y = fe.extract_multiclass_head(instance_grd)
                    Y.append(grd_y)

            X = np.asarray(X)
            Y = np.asarray(Y)
            print X.shape, Y.shape

if __name__ == "__main__":
    main()
