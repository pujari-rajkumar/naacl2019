import sys
import os
from drail.parser import parser


def main():

    filename = sys.argv[1]
    text = open(filename).read()
    par = parser.Parser()
    par.build()
    par.parse(filename)

    # Uncomment to see output of parser
    print "\nentities", par.entity_arguments
    print "\npredicates", par.predicate_entities, par.predicate_arguments
    print "\nfiles", par.files
    print "\ngroupby", par.groupby
    print "\nfeat_file:", par.femodule
    print "\nfeat_class:", par.feclass
    print "\ndb_file:", par.dbmodule
    print "\ndb_class:", par.dbclass
    print "\nsc_file:", par.scmodule


    for i, ruleset in enumerate(par.rulesets):
        print "\nRuleset # {0}\n".format(i)

        if 'rule' in ruleset:
            print "Rules"
            for rule in ruleset['rule']:
                print rule
                print "\tfilters:", rule.filters
                print "\tfeat_functions:", rule.feat_functions
                print "\tdb_func:", rule.dbfunc
                print "\tsc_func:", rule.scoring_function
                print "\tlambda:", rule.lambda_
                print "\ttarget:", rule.target
                print "\tsplit_on:", rule.split_on
                print "\tsplit_on_type:", rule.split_on_ttype
        else:
            print "No rules parsed"

        if 'constr' in ruleset:
            print "\nConstraints"
            for rule in ruleset['constr']:
                print rule
                print "\tfilters:", rule.filters
        else:
            print "No constraints parsed"
if __name__ == "__main__":
    main()
