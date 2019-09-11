import sys
import os
from drail.database import Database
from drail.parser import parser

def main():
    if os.path.isfile(":memory"):
        os.system("rm :memory")

    if len(sys.argv) != 3:
        sys.exit("Usage: python {0} [rule_file] [data_path]".format(sys.argv[0]))

    filename = sys.argv[1]
    path = sys.argv[2]
    text = open(filename).read()
    par = parser.Parser()
    par.build()
    par.parse(filename)

    # Uncomment to see output of parser
    print "predicates", par.predicate_arguments
    print "labels", par.label_types.keys()
    print "files", par.files
    print "rulesets", par.rulesets
    print "groupby", par.groupby

    db = Database()

    db.load_predicates(par.predicate_arguments, path, par.files)
    db.load_labels(par.label_types.keys(), path, par.files)
    print db.table_index

    for ruleset, group_by in zip(par.rulesets, par.groupby):
        instances = db.get_ruleset_instances(ruleset, group_by)

        for i in instances:
            instance_groundings = []
            print "\ninstance " + str(i)
            for rule_template in ruleset:

                if not rule_template.head.isobs:
                    '''if the predicate is not observed we have a binary pred problem'''
                    # this will be positive examples
                    rule_groundings = db.unfold_train_groundings(rule_template,
                                                                 group_by=group_by,
                                                                 instance_id=i)
                    # this will be negative examples
                    neg_rule_groundings = db.unfold_train_groundings(rule_template,
                                                                     neg_head=True,
                                                                     group_by=group_by,
                                                                     instance_id=i)

                    n_pos = len(rule_groundings)
                    n_neg = len(neg_rule_groundings)

                    print "Pos examples", n_pos / (n_pos + 1.0 * n_neg) * 100
                    print "Neg examples", n_neg / (n_pos + 1.0 * n_neg) * 100

                elif rule_template.head.label_type() is not None:
                    '''if the predicate is observed, then we must have an unobserved variable'''
                    rule_groundings = db.unfold_train_groundings(rule_template,
                                                                 group_by=group_by,
                                                                 instance_id=i)
                else:
                    '''if the predicate is observed and the variables are observed
                    we don't have anything to predict'''
                    print "This rule has nothing to predict"
                    continue


            for i in range(0, min(100, len(rule_groundings))):
                print rule_groundings[i]

    os.system("rm :memory")

if __name__ == "__main__":
    main()



