import sys
import os
from drail.database import Database
from drail.parser import parser
from drail.inference.ILPInferencer import *
from collections import OrderedDict


def main():
    print "main"
    if os.path.isfile(":memory"):
        os.system("rm :memory")

    if len(sys.argv) != 2:
        sys.exit("Usage:" + " python " + sys.argv[0] +" [rule file]")

    #'''
    # fout = open("testresult.txt", 'w')
    filename = sys.argv[1]
    text = open(filename).read()
    par = parser.Parser()
    par.build()
    par.parse(filename)

    # Uncomment to see output of parser
    print "predicates", par.predicate_arguments
    print "files", par.files
    print "rulesets", par.rulesets
    print "groupby", par.groupby

    db = Database()
    # Use your own path

    path = "examples/EntityRelation/all"

    db.load(par.predicate_arguments, path, par.files)
    print db.table_index

    for ruleset, group_by in zip(par.rulesets, par.groupby):
        # this will return the ids that can then be used to query the database
        # to group ILP problems
        instances = db.get_ruleset_instances(ruleset, group_by)
        #print instances

        # fout.write(str(instances) + "\n")

        for sent_id in instances:
            instance_groundings = []
            for rule_template in ruleset:
                rule_groundings = db.unfold_rule_groundings(rule_template, group_by, sent_id)
                # fout.write(str(rule_groundings) + "\n")
                for g in rule_groundings:
                    print g
                instance_groundings += rule_groundings
            # instantiate an ILPinferencer
            ruleWeights = OrderedDict()
            for ruleGrg in instance_groundings:
                ruleWeights[ruleGrg] = 0.5
            inferencer = ILPInferencer(ruleWeights, 'LP')
            inferencer.createModel("testModel {}".format(sent_id))

            inferencer.addVariables()
            inferencer.addConstraints()

            print(str(inferencer.model))
            inferencer.optimize()
            predVal = inferencer.getAttriValues()

            break
            
    #    fout.write("\n")
    # fout.close()
    #'''

    os.system("rm :memory")


if __name__ == "__main__":
    main()

