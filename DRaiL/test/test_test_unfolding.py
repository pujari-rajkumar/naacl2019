import sys
import os
from drail.database import Database
from drail.parser import parser
from drail.model.target import TargetType


def main():
    if len(sys.argv) < 3:
        sys.exit("Usage:" + " python " + sys.argv[0] +" [rule file] [data path] [filter_file1, filter_file2, ...]")
    #fout = open("testresult.txt", 'w')

    filename = sys.argv[1]
    text = open(filename).read()
    par = parser.Parser()
    par.build()
    par.parse(filename)

    filter_instances = []
    for i in range(3, len(sys.argv)):
        with open(sys.argv[i]) as f:
            for line in f:
                filter_instances.append(line.strip())

    # Uncomment to see output of parser
    print "predicates", par.predicate_arguments
    print "labels", par.label_types.keys()
    print "files", par.files
    print "rulesets", par.rulesets
    print "groupby", par.groupby

    db = Database()
    # Use your own path

    path = sys.argv[2]

    db.load_predicates(par.predicate_arguments, path, par.files)
    db.load_labels(par.label_types.keys(), path, par.files)
    print db.table_index

    ruleset = par.rulesets[0]
    groupby = par.groupby[0]

    for i, rule_template in enumerate(ruleset['predict']):
        #fout.write("\nrule_template: " + str(i))

        # this will return the ids that can then be used to query the database
        # to group ILP problems
        print "\ntemplate", rule_template

        '''
        if the predicate is not observed we have a binary predicate problem
        '''
        if rule_template.head.isobs == False:
            # this will be positive examples
            rule_groundings = db.unfold_train_groundings(rule_template, filter_by=(groupby, filter_instances))
            # this will be negative examples
            neg_rule_groundings = db.unfold_train_groundings(rule_template, filter_by=(groupby, filter_instances), neg_head=True)

            n_pos = len(rule_groundings)
            n_neg = len(neg_rule_groundings)


            print "Pos examples", n_pos / (n_pos + 1.0 * n_neg) * 100
            print "Neg examples", n_neg / (n_pos + 1.0 * n_neg) * 100
        elif rule_template.head.label_type() is not None:
            '''
            if the predicate is observed, then we must have an unobserved variable
            '''
            rule_groundings = db.unfold_train_groundings(rule_template, filter_by=(groupby, filter_instances))
        else:
            '''
            if the predicate is observed and the variables are observed
            we don't have anything to predict
            '''
            print "This rule has nothing to predict"
            continue

        if rule_template.head.isobs == False:
            print "NEG"
            for i in range(0, 100):
                print neg_rule_groundings[i]
            print "\nPOS"
        for i in range(0, min(len(rule_groundings), 100)):
            print rule_groundings[i]


        #fout.write(str(rule_groundings) + "\n")
        #fout.write("\n")
#fout.close()


if __name__ == "__main__":
    main()

