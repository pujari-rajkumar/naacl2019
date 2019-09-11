import sys
import os
import imp
from drail.database import Database
from drail.parser import parser
from drail.model.rule import RuleGrounding

def main():
    if len(sys.argv) < 3:
        sys.exit("Usage:" + " python " + sys.argv[0] +" [rule file] [data path] [dbmodulepath (optional)]")
    #fout = open("testresult.txt", 'w')

    filename = sys.argv[1]
    path = sys.argv[2]
    if len(sys.argv) > 3:
        dbmodule_path = sys.argv[3]

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

    ruleset = par.rulesets[0]
    group_by = par.groupby[0]

    if par.dbmodule is None:
        db = Database()
        db.load_predicates(par.predicate_arguments,
                           path, par.files)
        db.load_predicates(par.entity_arguments,
                           path, par.files)
        db.load_labels(par.label_types.keys(),
                       path, par.files)
        db.predicate_entities = par.predicate_entities
        print db.table_index
    else:
        module_path = os.path.join(dbmodule_path, "{0}.py".format(dbmodule))
        mod = imp.load_source(par.dbmodule, module_path)
        db_class = getattr(mod, par.dbclass)
        db = db_class()
        db.load_data(dataset_path)

    # this will return the ids that can then be used to query the database
    # to group ILP problems
    instances = db.get_ruleset_instances(ruleset['rule'], group_by, [], False)

    for i in instances:
        instance_groundings = []; constraint_groundings = []
        if i != 17:
            continue

        #fout.write("\ninstance: " + str(i))
        print "\ninstance " + str(i)

        gold_heads = db.get_gold_predicates(ruleset['rule'], [],
                                            group_by, i)
        gold_heads_set = set([RuleGrounding.pred_str(head) for head in gold_heads])
	#print gold_heads_set

        for ruleidx, rule_template in enumerate(ruleset['rule']):
            print "Rule", rule_template
            class_split = None

            if par.dbmodule is None:
                rule_groundings = db.unfold_rule_groundings(
                        rule_template, ruleidx, group_by, i, class_split,
                        gold_heads_set, filter_by=[])
            else:
                rule_groundings = \
                    getattr(db, rule_template.dbfunc)(
                            istrain=False,
                            isneg=False,
                            filters=[],
                            split_class=class_split,
                            instance_id=i)
            #fout.write(str(rule_groundings) + "\n")
            instance_groundings += rule_groundings

        for rule_template in ruleset['constr']:
            print "Constraint", rule_template
            class_split = None

            if par.dbmodule is None:
                rule_groundings = db.unfold_rule_groundings(
                        rule_template, ruleidx, group_by, i, class_split,
                        gold_heads_set, filter_by=[])
            else:
                rule_groundings = \
                    getattr(db, rule_template.dbfunc)(
                            istrain=False,
                            isneg=False,
                            filters=[],
                            split_class=class_split,
                            instance_id=i)
            #fout.write(str(rule_groundings) + "\n")
            constraint_groundings += rule_groundings

        print "Instance groundings"
        for gr in instance_groundings:
            print gr


        print "Constraint groundings"
        for gr in constraint_groundings:
            print gr
        break
        #fout.write("\n")
    #fout.close()


if __name__ == "__main__":
    main()

