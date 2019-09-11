import sqlite3
import os
from model.rule import RuleGrounding
from model.predicate import PredicateTemplate
from model.argument import ArgumentType
from model.argument import sqlite_type

class Database(object):

    def __init__(self):
        self.conn = sqlite3.connect(":memory:")
        self.cur = self.conn.cursor()
        self.table_index = {}
        self.table_index_types = {}
        self.predicate_entities = {}

    def drop_table(self, table_name):
        query = "DROP TABLE IF EXISTS {0}".format(table_name)
        self.cur.execute(query)

    def create_indexes(self, table_name, columns):
        for col in columns:
            query = "CREATE INDEX {0}_{1} ON {0} ({1})"
            query = query.format(table_name, col)
            #print query
            self.cur.execute(query)

    def create_table_predicate(self, table_name, arguments):
        self.table_index[table_name] = [a.name for a in arguments]
        self.table_index_types[table_name] = [a.typ for a in arguments]
        args = ", ".join([a.name + " " + a.sqlite_type() for a in arguments])
        indexes = [a.name for a in arguments if a.typ == ArgumentType.UniqueID or a.typ == ArgumentType.UniqueString]

        query = "CREATE TABLE {0} ({1})"
        query = query.format(table_name, args)
        #print query
        self.cur.execute(query)
        self.create_indexes(table_name, indexes)

    # OJO: right now assuming all labels are categorical
    def create_table_label(self, table_name):
        self.table_index[table_name] = ["label"]
        query = "CREATE TABLE {0} (label TEXT)"
        query = query.format(table_name)
        # print query
        self.cur.execute(query)
        self.create_indexes(table_name, ["label"])

    def create_table_filter(self, table_name, args, arg_types, arg_data):
        self.table_index[table_name] = [a for a in args]
        self.table_index_types[table_name] = [t for t in arg_types]
        args_ = ", ".join([a + " " + sqlite_type(t) for (a,t) in zip(args, arg_types)])
        query = "CREATE TABLE {0} ({1})"
        query = query.format(table_name, args_)
        # print query
        self.cur.execute(query)

    def add_column(self, table_name, column_name, column_type, default):
        query = "ALTER TABLE {0} ADD COLUMN {1} {2} DEFAULT {3}"
        query = query.format(table_name, column_name, column_type, default)
        #print query
        self.cur.execute(query)

    def insert_row(self, table_name, values):
        args = self.table_index[table_name]
        args = ", ".join([a for a in args])
        values = [v.replace('"', "'") for v in values]
        values = ", ".join(['"' + v + '"' for v in values])
        query = "INSERT INTO {0} ({1}) VALUES ({2})"
        query = query.format(table_name, args, values)
        self.cur.execute(query)

    def load_data(self, path, parser):
        self.load_predicates(parser.predicate_arguments,
                             path, parser.files)
        self.load_predicates(parser.entity_arguments,
                             path, parser.files)
        self.predicate_entities = parser.predicate_entities

    def load_predicates(self, predicate_arguments, path, files):
        # create all.name tables
        for pred in predicate_arguments:
            arguments = predicate_arguments[pred]
            fullpath = os.path.join(path, files[pred])
            with open(fullpath) as f:
                name = files[pred].split(".")[0]
                self.create_table_predicate(pred, arguments)

                content = f.readlines()
                for line in content:
                    values = line.strip().split()
                    self.insert_row(pred, values)

    def load_labels(self, labels, path, files):
        for label in labels:
            fullpath = os.path.join(path, files[label])
            with open(fullpath) as f:
                name = files[label].split(".")[0]
                self.create_table_label(label)
                content = f.readlines()
                for line in content:
                    values = line.strip().split()
                    self.insert_row(label, values)

    def add_filters(self, filters):
        for (filter_table, filter_col, filter_id, filter_values) in filters:
            self.add_column(filter_table, filter_col, 'INTEGER', 0)
            self.create_indexes(filter_table, [filter_col])
            for value in filter_values:
                query = "UPDATE {0} SET {1} = 1 WHERE {2} = {3}"
                query = query.format(filter_table, filter_col, filter_id, value)
                self.cur.execute(query)

    # OJO: this code is super messy, i need to rewrite it
    # but it works
    def select_train_grounding(self, predicate_templates, join_vars, filter_by, group_by, group_id,
                               split_on_var, split_on_value):
        seen = {}
        pred_tables = {}
        group_by_aliases = []

        if group_by is not None:
            group_by_col = self.table_index[group_by[0]][int(group_by[1]) - 1]

        query = "SELECT * FROM {0} p0 ".format(predicate_templates[0].name)
        for k, var in enumerate(predicate_templates[0].variables):

            if var.arg not in seen:
                seen[var.arg] = [(0, k)]
            else:
                seen[var.arg].append((0, k))
            pred_tables["p0"] = predicate_templates[0].name

        if group_by is not None and predicate_templates[0].name == group_by[0]:
            group_by_aliases.append(("p0", group_by_col))

        constant_vars = []
        negations = []
        last_i = 0

        for i in range(1, len(predicate_templates)):
            if predicate_templates[i].isneg:
                negations.append(predicate_templates[i])
                continue
            t = predicate_templates[i].name
            alias = "p{0}".format(i)

            if group_by is not None and t == group_by[0]:
                group_by_aliases.append((alias, group_by_col))

            pred_tables[alias] = t
            query += "INNER JOIN {0} p{1} ".format(t, i)

            first = True
            for j, var in enumerate(predicate_templates[i].variables):
                if var.isconstant:
                    cond = "p{0}.{1}".format(i, self.table_index[t][j])
                    constant_vars.append((cond, var.arg))
                elif var.arg in seen:
                    j2 = seen[var.arg][0][1]
                    alias = "p{0}".format(seen[var.arg][0][0])
                    t2 = pred_tables[alias]

                    cond1 = "p{0}.{1}".format(i, self.table_index[t][j])
                    cond2 = "p{0}.{1}".format(seen[var.arg][0][0],
                                              self.table_index[t2][j2])

                    if first:
                        query += "ON {0} = {1} ".format(cond1, cond2)
                        first = False
                    else:
                        query += "AND {0} = {1} ".format(cond1, cond2)

                if var.arg not in seen:
                    seen[var.arg] = [(i, j)]
                else:
                    seen[var.arg].append((i,j))
                last_i = i


        last_i += 1

        for var in join_vars:
            if var not in seen:
                continue

            seen_ = seen[var][0]
            j = seen_[1]
            alias = "p{0}".format(seen_[0])
            t = pred_tables[alias]
            t2 = self.predicate_entities[t][j]
            j2 = 0

            query += "INNER JOIN {0} p{1} ".format(t2, last_i)

            cond1 = "p{0}.{1}".format(last_i, self.table_index[t2][j2])
            cond2 = "p{0}.{1}".format(seen_[0], self.table_index[t][j])

            query += "ON {0} = {1} ".format(cond1, cond2)
            seen[var].append((last_i,j2))


            last_i += 1


        first = True


        for var, col, value in filter_by:
            if var not in seen:
                continue
            #print var, col, value

            cond = "p{0}.{1}".format(seen[var][-1][0], col)
            if first:
                query += "WHERE {0} = {1} ".format(cond, value)
                first = False
            else:
                query += "AND {0} = {1} ".format(cond, value)

        if len(negations) > 0:
            first_inner = True
            for j, neg_pred in enumerate(negations):

                neg_alias = "n{0}".format(j)
                inner_select = "SELECT NULL FROM {0} {1} ".format(neg_pred.name, neg_alias)
                for var in neg_pred.variables:
                    j2 = seen[var.arg][0][1]
                    alias = "p{0}".format(seen[var.arg][0][0])
                    t2 = pred_tables[alias]
                    cond1 = "{0}.{1}".format(alias, self.table_index[t2][j2])
                    cond2 = "{0}.{1}".format(neg_alias, self.table_index[neg_pred.name][j2])
                    if first_inner:
                        inner_select += "WHERE {0} = {1} ".format(cond2, cond1)
                        first_inner = False
                    else:
                        inner_select += "AND {0} = {1} ".format(cond2, cond1)
            query += "WHERE NOT EXISTS ({0}) ".format(inner_select)
            first = False

        for const in constant_vars:
            if first:
                query += "WHERE {0} = '{1}' ".format(const[0], const[1])
                first = False
            else:
                query += "AND {0} = '{1}' ".format(const[0], const[1])

        for x in range(len(group_by_aliases)):
            alias = group_by_aliases[x][0]
            cond = alias + "." + group_by_aliases[x][1]
            if first:
                query += "WHERE {0} = {1} ".format(cond, group_id)
                first = False
            else:
                 query += "AND {0} = {1} ".format(cond, group_id)

        if split_on_var is not None and split_on_value is not None:
            j = seen[split_on_var][0][1]
            alias = "p{0}".format(seen[split_on_var][0][0])
            t = pred_tables[alias]
            if first:
                query += " WHERE {0}.{1} == '{2}'".format(
                        alias, self.table_index[t][j], split_on_value)
                first = False
            else:
                query += " AND {0}.{1} == '{2}'".format(
                        alias, self.table_index[t][j], split_on_value)
        #print "\nSELECT TRAIN\n", query
        self.cur.execute(query)
        #print "."
        return self.cur.fetchall()

    # OJO: this code is super messy, i need to rewrite it
    # but it works
    def select_train_grounding_neg_head(self, body_templates, head_template,
                                        join_vars, filter_by, group_by, group_id,
                                        split_on_var, split_on_value):
        seen = {}
        pred_tables = {}
        group_by_aliases = []

        query = "SELECT * FROM {0} p0 ".format(body_templates[0].name)
        for k, var in enumerate(body_templates[0].variables):
            if var.arg not in seen:
                seen[var.arg] = [(0, k)]
            else:
                seen[var.arg].append((0, k))
            pred_tables["p0"] = body_templates[0].name
        if group_by is not None and body_templates[0].name == group_by[0]:
            group_by_col = self.table_index[group_by[0]][int(group_by[1]) - 1]
            group_by_aliases.append(("p0", group_by_col))

        constant_vars = []
        negations = []
        last_i = 0

        for i in range(1, len(body_templates)):
            if body_templates[i].isneg:
                negations.append(body_templates[i])
                continue
            t = body_templates[i].name
            alias = "p{0}".format(i)

            if group_by is not None and t == group_by[0]:
                group_by_col = self.table_index[group_by[0]][int(group_by[1]) - 1]
                group_by_aliases.append((alias, group_by_col))

            pred_tables[alias] = t
            query += "INNER JOIN {0} p{1} ".format(t, i)

            first = True
            for j, var in enumerate(body_templates[i].variables):
                if var.isconstant:
                    cond = "p{0}.{1}".format(i, self.table_index[t][j])
                    constant_vars.append((cond, var.arg))
                elif var.arg in seen:
                    j2 = seen[var.arg][0][1]
                    alias = "p{0}".format(seen[var.arg][0][0])
                    t2 = pred_tables[alias]

                    cond1 = "p{0}.{1}".format(i, self.table_index[t][j])
                    cond2 = "p{0}.{1}".format(seen[var.arg][0][0],
                                              self.table_index[t2][j2])

                    if first:
                        query += "ON {0} = {1} ".format(cond1, cond2)
                        first = False
                    else:
                        query += "AND {0} = {1} ".format(cond1, cond2)

                if var.arg not in seen:
                    seen[var.arg] = [(i, j)]
                else:
                    seen[var.arg].append((i,j))

        i = len(body_templates)
        t = head_template.name
        alias = "p{0}".format(i)
        pred_tables[alias] = t

        query += "LEFT OUTER JOIN {0} p{1} ".format(t, i)
        first = True
        null_vars = []

        for j, var in enumerate(head_template.variables):
            if var.isconstant:
                cond1 = "p{0}.{1}".format(i, self.table_index[t][j])
                cond2 = "'{0}'".format(var.arg)
            elif var.arg in seen:
                j2 = seen[var.arg][0][1]
                alias = "p{0}".format(seen[var.arg][0][0])
                t2 = pred_tables[alias]

                cond1 = "p{0}.{1}".format(i, self.table_index[t][j])
                cond2 = "p{0}.{1}".format(seen[var.arg][0][0],
                                          self.table_index[t2][j2])
                null_vars.append(cond1)

                if first:
                    query += "ON {0} = {1} ".format(cond1, cond2)
                    first = False
                else:
                    query += "AND {0} = {1} ".format(cond1, cond2)

            if var.arg not in seen:
                seen[var.arg] = [(i,j)]
            else:
                seen[var.arg].append((i,j))

            if var.arg == split_on_var and split_on_value is not None:
                j = seen[split_on_var][0][1]
                alias = "p{0}".format(seen[split_on_var][0][0])
                t = pred_tables[alias]
                if first:
                    query += "ON {0}.{1} == '{2}' ".format(
                            alias, self.table_index[t][j], split_on_value)
                    first = False
                else:
                    query += "AND {0}.{1} == '{2}' ".format(
                            alias, self.table_index[t][j], split_on_value)

        last_i = i
        #print join_vars
        last_i += 1

        for var in join_vars:
            if var not in seen:
                continue

            seen_ = seen[var][0]
            j = seen_[1]
            alias = "p{0}".format(seen_[0])
            t = pred_tables[alias]
            t2 = self.predicate_entities[t][j]
            j2 = 0

            query += "INNER JOIN {0} p{1} ".format(t2, last_i)

            cond1 = "p{0}.{1}".format(last_i, self.table_index[t2][j2])
            cond2 = "p{0}.{1}".format(seen_[0], self.table_index[t][j])

            query += "ON {0} = {1} ".format(cond1, cond2)
            seen[var].append((last_i,j2))


            last_i += 1


        first = True


        for var, col, value in filter_by:
            if var not in seen:
                continue
            #print var, col, value

            cond = "p{0}.{1}".format(seen[var][-1][0], col)
            if first:
                query += "WHERE {0} = {1} ".format(cond, value)
                first = False
            else:
                query += "AND {0} = {1} ".format(cond, value)

        if len(negations) > 0:
            first_inner = True
            for j, neg_pred in enumerate(negations):

                neg_alias = "n{0}".format(j)
                inner_select = "SELECT NULL FROM {0} {1} ".format(neg_pred.name, neg_alias)
                for var in neg_pred.variables:
                    j2 = seen[var.arg][0][1]
                    alias = "p{0}".format(seen[var.arg][0][0])
                    t2 = pred_tables[alias]
                    cond1 = "{0}.{1}".format(alias, self.table_index[t2][j2])
                    cond2 = "{0}.{1}".format(neg_alias, self.table_index[neg_pred.name][j2])
                    if first_inner:
                        inner_select += "WHERE {0} = {1} ".format(cond2, cond1)
                        first_inner = False
                    else:
                        inner_select += "AND {0} = {1} ".format(cond2, cond1)
            query += "WHERE NOT EXISTS ({0}) ".format(inner_select)
            first = False

        for const in constant_vars:
            if first:
                query += "WHERE {0} = '{1}' ".format(const[0], const[1])
                first = False
            else:
                query += "AND {0} = '{1}' ".format(const[0], const[1])
        for null in null_vars:
            if first:
                query += "WHERE {0} IS NULL ".format(null)
                first = False
            else:
                query += "AND {0} IS NULL ".format(null)

        for x in range(len(group_by_aliases)):
            alias = group_by_aliases[x][0]
            cond = alias + "." + group_by_aliases[x][1]
            if first:
                query += "WHERE {0} = {1} ".format(cond, group_id)
                first = False
            else:
                query += "AND {0} = {1} ".format(cond, group_id)


        #print "\nSELECT TRAIN NEG\n", query
        self.cur.execute(query)
        return self.cur.fetchall()

    def select_target_counts(self, predicate_templates, rule_target,
                            join_vars, filter_by,
                            group_by, group_id):
        seen = {}
        pred_tables = {}
        group_by_aliases = []
        group_by_col = self.table_index[group_by[0]][int(group_by[1]) - 1]


        select = "SELECT {0}, COUNT({0}) "
        query = "FROM {0} p0 ".format(predicate_templates[0].name)
        for k, var in enumerate(predicate_templates[0].variables):
            if var.arg not in seen:
                seen[var.arg] = [(0, k)]
            else:
                seen[var.arg].append((0, k))
            pred_tables["p0"] = predicate_templates[0].name

        if group_by is not None and predicate_templates[0].name == group_by[0]:
            group_by_aliases.append(("p0", group_by_col))

        constant_vars = []
        negations = []
        last_i = 0

        for i in range(1, len(predicate_templates)):
            if predicate_templates[i].isneg:
                negations.append(predicate_templates[i])
                continue
            t = predicate_templates[i].name
            alias = "p{0}".format(i)

            if group_by is not None and t == group_by[0]:
                group_by_aliases.append((alias, group_by_col))

            pred_tables[alias] = t
            query += "INNER JOIN {0} p{1} ".format(t, i)

            first = True
            for j, var in enumerate(predicate_templates[i].variables):
                if var.isconstant:
                    cond = "p{0}.{1}".format(i, self.table_index[t][j])
                    constant_vars.append((cond, var.arg))
                elif var.arg in seen:
                    j2 = seen[var.arg][0][1]
                    alias = "p{0}".format(seen[var.arg][0][0])
                    t2 = pred_tables[alias]

                    cond1 = "p{0}.{1}".format(i, self.table_index[t][j])
                    cond2 = "p{0}.{1}".format(seen[var.arg][0][0],
                                              self.table_index[t2][j2])

                    if first:
                        query += "ON {0} = {1} ".format(cond1, cond2)
                        first = False
                    else:
                        query += "AND {0} = {1} ".format(cond1, cond2)

                if var.arg not in seen:
                    seen[var.arg] = [(i, j)]
                else:
                    seen[var.arg].append((i,j))
                last_i = i

        #print join_vars
        last_i += 1

        for var in join_vars:
            if var not in seen:
                continue

            seen_ = seen[var][0]
            j = seen_[1]
            alias = "p{0}".format(seen_[0])
            t = pred_tables[alias]
            t2 = self.predicate_entities[t][j]
            j2 = 0

            query += "INNER JOIN {0} p{1} ".format(t2, last_i)

            cond1 = "p{0}.{1}".format(last_i, self.table_index[t2][j2])
            cond2 = "p{0}.{1}".format(seen_[0], self.table_index[t][j])

            query += "ON {0} = {1} ".format(cond1, cond2)
            seen[var].append((last_i,j2))


            last_i += 1


        first = True


        for var, col, value in filter_by:
            if var not in seen:
                continue
            #print var, col, value

            cond = "p{0}.{1}".format(seen[var][-1][0], col)
            if first:
                query += "WHERE {0} = {1} ".format(cond, value)
                first = False
            else:
                query += "AND {0} = {1} ".format(cond, value)

        if len(negations) > 0:
            first_inner = True
            for j, neg_pred in enumerate(negations):

                neg_alias = "n{0}".format(j)
                inner_select = "SELECT NULL FROM {0} {1} ".format(neg_pred.name, neg_alias)
                for var in neg_pred.variables:
                    j2 = seen[var.arg][0][1]
                    alias = "p{0}".format(seen[var.arg][0][0])
                    t2 = pred_tables[alias]
                    cond1 = "{0}.{1}".format(alias, self.table_index[t2][j2])
                    cond2 = "{0}.{1}".format(neg_alias, self.table_index[neg_pred.name][j2])
                    if first_inner:
                        inner_select += "WHERE {0} = {1} ".format(cond2, cond1)
                        first_inner = False
                    else:
                        inner_select += "AND {0} = {1} ".format(cond2, cond1)
            query += "WHERE NOT EXISTS ({0}) ".format(inner_select)
            first = False

        for const in constant_vars:
            if first:
                query += "WHERE {0} = '{1}' ".format(const[0], const[1])
                first = False
            else:
                query += "AND {0} = '{1}' ".format(const[0], const[1])

        for x in range(len(group_by_aliases)):
            alias = group_by_aliases[x][0]
            cond = alias + "." + group_by_aliases[x][1]
            if first:
                query += "WHERE {0} = {1} ".format(cond, group_id)
                first = False
            else:
                 query += "AND {0} = {1} ".format(cond, group_id)

        # add the select statement
        j = seen[rule_target][0][1]
        alias = "p{0}".format(seen[rule_target][0][0])
        t = pred_tables[alias]
        column = "{0}.{1}".format(alias, self.table_index[t][j])

        query += "GROUP BY {0}".format(column)

        select = select.format(column)
        query = select + query
        #print query
        self.cur.execute(query)
        ret =  self.cur.fetchall()
        return ret


    # OJO: this code is super messy, i need to rewrite it
    # but it works
    def select_grounding(self, predicate_templates, target_vars,
                         target_preds, join_vars,
                         filter_by, group_by, group_id, isconstr,
                         split_on_var, split_on_value, known_variables,
                         unknown_variables):
        seen = {}
        pred_tables = {}
        group_by_aliases = []
        if group_by is not None:
            group_by_col = self.table_index[group_by[0]][int(group_by[1]) - 1]

        #print "PREDICATE TEMPLATES", predicate_templates
        query = "SELECT * FROM {0} p0 ".format(predicate_templates[0].name)
        # I am assuming we want to order by the first predicate for now
        # we might want to specify this in the rule file
        order_by = predicate_templates[0]

        for k, var in enumerate(predicate_templates[0].variables):
            if var.arg not in seen:
                seen[var.arg] = [(0, k)]
            else:
                seen[var.arg].append((0, k))
            pred_tables["p0"] = predicate_templates[0].name
        if group_by is not None and predicate_templates[0].name == group_by[0]:
            group_by_aliases.append(("p0", group_by_col))

        negations = []
        constant_vars = []
        last_i = 0

        #print "TARGET PREDS", target_preds
        for i in range(1, len(predicate_templates)):
            last_i = i


            t = predicate_templates[i].name

            # do not query target predicates or constraint heads
            if target_preds[i]:
                for j, var in enumerate(predicate_templates[i].variables):
                    if var.arg not in seen:
                        alias = "p{0}".format(i)
                        pred_tables[alias] = t
                        seen[var.arg] = [(i,j)]
                continue

            if predicate_templates[i].isneg:
                negations.append(predicate_templates[i])
                continue

            alias = "p{0}".format(i)

            if group_by is not None and t == group_by[0]:
                group_by_aliases.append((alias, group_by_col))

            pred_tables[alias] = t
            query += "INNER JOIN {0} p{1} ".format(t, i)

            first = True
            for j, var in enumerate(predicate_templates[i].variables):
                if var.isconstant:
                    cond = "p{0}.{1}".format(i, self.table_index[t][j])
                    constant_vars.append((cond, var.arg))

                if var.arg in seen and var.arg not in target_vars:
                    j2 = seen[var.arg][0][1]
                    alias = "p{0}".format(seen[var.arg][0][0])
                    t2 = pred_tables[alias]

                    cond1 = "p{0}.{1}".format(i, self.table_index[t][j])
                    cond2 = "p{0}.{1}".format(seen[var.arg][0][0],
                                              self.table_index[t2][j2])

                    if first:
                        query += "ON {0} = {1} ".format(cond1, cond2)
                        first = False
                    else:
                        query += "AND {0} = {1} ".format(cond1, cond2)

                if var.arg not in seen:
                    seen[var.arg] = [(i, j)]
                else:
                    seen[var.arg].append((i,j))


        last_i += 1

        temp = set([]); temp_list = []
        for var in unknown_variables:
            if var in known_variables:
                continue
            if var in temp:
                continue

            seen_ = seen[var][0]
            alias = "p{0}".format(seen_[0])

            t = pred_tables[alias]
            j = seen_[1]
            t2 = self.predicate_entities[t][j]
            j2 = 0

            query += "INNER JOIN {0} p{1} ".format(t2, last_i)
            seen[var].append((last_i,j2))
            pred_tables["p{0}".format(last_i)] = t2

            last_i += 1
            temp.add(var)
            temp_list.append(var)
        #print query

        join_vars = join_vars & (known_variables)
        for var in join_vars:
            seen_ = seen[var][0]
            alias = "p{0}".format(seen_[0])

            t = pred_tables[alias]
            j = seen_[1]
            t2 = self.predicate_entities[t][j]
            j2 = 0

            query += "INNER JOIN {0} p{1} ".format(t2, last_i)
            cond1 = "p{0}.{1}".format(last_i, self.table_index[t2][j2])
            cond2 = "p{0}.{1}".format(seen_[0], self.table_index[t][j])

            query += "ON {0} = {1} ".format(cond1, cond2)

            seen[var].append((last_i,j2))
            pred_tables["p{0}".format(last_i)] = t2

            last_i += 1


        #print query
        first = True


        for var, col, value in filter_by:
            if var not in seen:
                continue
            #print var, col, value

            cond = "p{0}.{1}".format(seen[var][-1][0], col)
            if first:
                query += "WHERE {0} = {1} ".format(cond, value)
                first = False
            else:
                query += "AND {0} = {1} ".format(cond, value)

        for const in constant_vars:
            if first:
                query += "WHERE {0} = '{1}' ".format(const[0], const[1])
                first = False
            else:
                query += "AND {0} = '{1}' ".format(const[0], const[1])


        if len(group_by_aliases) > 0:
            cond = group_by_aliases[0][0] + "." + \
                   group_by_aliases[0][1]
            if first:
                query += "WHERE {0} = {1} ".format(cond, group_id)
                first = False
            else:
                query += "AND {0} = {1} ".format(cond, group_id)
            for x in range(1, len(group_by_aliases)):
                alias = group_by_aliases[x][0]
                cond = alias + "." + group_by_aliases[x][1]
                query += "AND {0} = {1} ".format(cond, group_id)

        if split_on_var is not None and split_on_value is not None:
            j = seen[split_on_var][-1][1]
            alias = "p{0}".format(seen[split_on_var][-1][0])
            t = pred_tables[alias]
            if first:
                query += " WHERE {0}.{1} == '{2}' ".format(
                        alias, self.table_index[t][j], split_on_value)
                first = False
            else:
                query += " AND {0}.{1} == '{2}' ".format(
                        alias, self.table_index[t][j], split_on_value)

        if len(negations) > 0:
            first_inner = True
            for j, neg_pred in enumerate(negations):

                neg_alias = "n{0}".format(j)
                inner_select = "SELECT NULL FROM {0} {1} ".format(neg_pred.name, neg_alias)
                for var in neg_pred.variables:
                    j2 = seen[var.arg][-1][1]
                    alias = "p{0}".format(seen[var.arg][-1][0])
                    t2 = pred_tables[alias]
                    cond1 = "{0}.{1}".format(alias, self.table_index[t2][j2])
                    cond2 = "{0}.{1}".format(neg_alias, self.table_index[neg_pred.name][j2])
                    if first_inner:
                        inner_select += "WHERE {0} = {1} ".format(cond2, cond1)
                        first_inner = False
                    else:
                        inner_select += "AND {0} = {1} ".format(cond2, cond1)
            if not first:
                query += "AND NOT EXISTS ({0}) ".format(inner_select)
            else:
                query += "WHERE NOT EXISTS ({0}) ".format(inner_select)
            first = False

        query += "ORDER BY p0.{0}".format(self.table_index[order_by.name][0])
        #print "UNFOLD"
        #print query
        self.cur.execute(query)

        return self.cur.fetchall()

    # OJO: asuming we are always filtering by a predicate with a single variable
    # last argument ignored, it is there to provide a hack for something else
    def get_ruleset_instances(self, ruleset, group_by, filter_by=[], is_train_global=False):

        rule_template = ruleset[0]
        variables = set([])
        tables = {}; columns = {}
        group_by_col = self.table_index[group_by[0]][int(group_by[1]) - 1]
        query = "SELECT DISTINCT {0} FROM {1} ".format(
                    group_by_col, group_by[0])
        for pred in rule_template.body:
            if pred.name == group_by[0]:
                for j, var in enumerate(pred.variables):
                    tables[var.arg] = self.predicate_entities[group_by[0]][j]
                    columns[var.arg] = self.table_index[group_by[0]][j]
                    variables.add(var.arg)

        for var, col, value in filter_by:
            if var in variables:
                query += "INNER JOIN {0} ".format(tables[var])
                query += " ON {0}.{1} = {2}.{3} ".format(
                        group_by[0], columns[var], tables[var],
                        self.table_index[tables[var]][0])
                query += " WHERE {0}.{1} = {2}".format(
                        tables[var], col, value)
        self.cur.execute(query)
        ret = self.cur.fetchall()
        ret = [r[0] for r in ret]
        return ret

    # OJO: this code is messy, i need to change it
    # but it works!
    def unfold_train_groundings(self, rule_template, uid_template,
                                split_on_value, filter_by=[],
                                neg_head=False, noconst=False,
                                group_by=None, instance_id=None):
        ret = []
        # print rule_template.body
        # print "***"

        predicate_templates = rule_template.body + [rule_template.head]

        #print "\nPREDICATE TEMPLATES: ", predicate_templates
        #print "\n"

        # print predicate_templates
        # asuming the target is always a predicate with a single variable
        null_vars = []
        join_vars = set([var for (var, _, _) in filter_by + rule_template.filters])

        if not neg_head:
            selected = self.select_train_grounding(
                    predicate_templates, join_vars,
                    filter_by + rule_template.filters, group_by, instance_id,
                    rule_template.split_on, split_on_value)
        else:
            selected = \
                    self.select_train_grounding_neg_head(
                            rule_template.body, rule_template.head, join_vars,
                            filter_by + rule_template.filters, group_by, instance_id,
                            rule_template.split_on, split_on_value)


        #print "selected", len(selected)

        if len(selected) > 0:
            null_vars = set([i for i in range(0, len(selected[0]))
                                 if selected[0][i] == None])

        for s in selected:
            new_rule = RuleGrounding([], None, uid_template, not rule_template.head.isobs, None)
            index = 0
            seen_vars = {}
            for predicate_templ in rule_template.body:
                obs=True
                ttype=None
                target_pos = None
                args = []
                for i, var in enumerate(predicate_templ.variables):
                    if var.isconstant:
                        args.append(var.arg)
                    elif index not in null_vars:
                        seen_vars[var.arg] = s[index]
                        args.append(s[index])
                    else:
                        args.append(seen_vars[var.arg])
                    if not predicate_templ.isneg:
                        index += 1
                '''
                new_predicate = PredicateGrounding(predicate_templ.name, args,
                                                   ttype,predicate_templ.isneg,
                                                   obs, target_pos)
                '''
                new_predicate = {'name': predicate_templ.name,
                                 'arguments': args, 'ttype': ttype,
                                 'obs': obs, 'isneg': predicate_templ.isneg,
                                 'target_pos': target_pos}
                #print new_predicate
		new_rule.body.append(new_predicate)

            # skipping filters
            #index += filter_skip

            args = []
            predicate_templ = rule_template.head
            head_neg = predicate_templ.isneg
            for var in predicate_templ.variables:
                if var.isconstant:
                    args.append(var.arg)
                elif index not in null_vars:
                    seen_vars[var.arg] = s[index]
                    args.append(s[index])
                elif index in null_vars and var.arg in seen_vars:
                    args.append(seen_vars[var.arg])
                elif index in null_vars and var.arg not in seen_vars and var.arg == rule_template.split_on and split_on_value is not None:
                    args.append(split_on_value)
                index += 1
            obs=False

            # using this now so that we can move fast
            # might be updated later
            ttype=rule_template.head.label_type()
            tpos=rule_template.head.label_pos()
            '''
            new_predicate = PredicateGrounding(predicate_templ.name, args,
                                               ttype, predicate_templ.isneg,
                                               obs, tpos)
            '''
            new_predicate = {'name': predicate_templ.name, 'arguments': args,
                             'ttype': ttype, 'obs': obs, 'isneg': neg_head,
                             'obs': obs, 'target_pos': tpos}
            new_rule.head = new_predicate

            new_rule.build_predicate_index()
            new_rule.build_body_predicate_dic()
            ret.append(new_rule)
            # print new_rule
        '''
        if len(ret) > 0:
            print "\tS: ", selected[0]
            print "\tR: ", ret[0]
        print len(ret)
        '''
        return ret


    def process_rule_template(self, rule_template):
        body_ranges = []
        body_targets = {}
        target_abs_pos = []
        abs_index = 0

        # compute body
        known_variables = []; unknown_variables = []
        for i, pred_templ in enumerate(rule_template.body):
            if not pred_templ.isobs:
                body_ranges.append(None)
                unknown_variables += [var.arg for var in pred_templ.variables]
                continue
            if pred_templ.isneg:
                continue

            known_variables += [var.arg for var in pred_templ.variables]
            num_vars = len(pred_templ.variables)
            body_ranges.append((abs_index, abs_index + num_vars))

            for j, var in enumerate(pred_templ.variables):
                if not var.isobs:
                    isconst = False; const_value = None
                    if var.isconstant:
                        isconst = True; const_value = var.arg
                    target_abs_pos.append((abs_index + j, isconst, const_value))
                    body_targets[i] = (var.label.l_type, j, isconst, const_value)

            abs_index += num_vars


        # compute head
        pred_templ = rule_template.head
        head_range = None
        head_target = None

        if pred_templ.isobs:
            num_vars = len(pred_templ.variables)
            head_range = (abs_index, abs_index + num_vars)

            for j, var in enumerate(pred_templ.variables):
                if not var.isobs:
                    isconst = False; const_value = None
                    if var.isconstant:
                        isconst = True; const_value = var.arg
                    head_target = (var.label.l_type, j, isconst, const_value)
                    target_abs_pos.append((abs_index + j, isconst, const_value))
                    unknown_variables.append(var.arg)
                else:
                    known_variables.append(var.arg)
        else:
            unknown_variables += [var.arg for var in pred_templ.variables]

        known_variables = set(known_variables)
        unknown_variables = set(unknown_variables) - known_variables

        return body_ranges, body_targets, head_range, head_target, target_abs_pos, \
               known_variables, unknown_variables


    def get_gold_predicates(self, rule_templates, filter_by,
                            group_by, instance_id):
        gold_data = set([]); ret = []
        executed_queries = set([])
        negations = []
        group_by_col = self.table_index[group_by[0]][int(group_by[1]) - 1]

        for rule_template in rule_templates:
            pred = rule_template.head

            join_vars = set([var for (var, _, _) in filter_by + rule_template.filters])
            join_vars = join_vars & set([var.arg for var in pred.variables])
            filter_by = [f for f in filter_by if f[0] in join_vars]

            # group by pred indicates inference instance
            group_by_preds = rule_template.predicate_index[group_by[0]]

            join_preds = group_by_preds

            # select head
            query = "SELECT * FROM {0} ".format(pred.name)
            # join on group by predicate
            pred_variables = [var.arg for var in pred.variables]

            #print "join_preds", join_preds
            group_by_aliases = []
            for j, join_pred in enumerate(join_preds):
                join_variables = []; join_variable_pos = []
                if join_pred.isneg:
                    neg_variables = [var.arg for var in join_pred.variables]
                    if len(set(neg_variables) & set(pred_variables)) > 0:
                        negations.append(join_pred)
                    continue

                join_variables = [var.arg for var in join_pred.variables]

                query += "INNER JOIN {0} p{1} ".format(join_pred.name, j)
                group_by_aliases.append("p{0}".format(j))
                joiner_vars = set(pred_variables) & set(join_variables)

                for var in joiner_vars:
                    index_pred = pred_variables.index(var)
                    index_join = join_variables.index(var)

                    query += "ON {0}.{1} = p{2}.{3} ".format(
                            pred.name,
                            self.table_index[pred.name][index_pred],
                            j,
                            self.table_index[join_pred.name][index_join])

            # join on filters
            for var in join_vars:

                var_pos = pred.var_pos[var]
                join_pred_name = self.predicate_entities[pred.name][var_pos]

                query += "INNER JOIN {0} ".format(join_pred_name)
                cond1 = "{0}.{1}".format(pred.name, self.table_index[pred.name][var_pos])
                cond2 = "{0}.{1}".format(join_pred_name, self.table_index[join_pred_name][0])
                query += "ON {0} = {1} ".format(cond1, cond2)

            first = True

            for var, col, value in filter_by:

                join_pred_name = self.predicate_entities[pred.name][var_pos]
                cond = "{0}.{1}".format(join_pred_name, col)
                if first:
                    query += "WHERE {0} = {1} ".format(cond, value)
                    first = False
                else:
                    query += "AND {0} = {1} ".format(cond, value)

            # filter on instance id
            for alias in group_by_aliases:
                if not first:
                    query += "AND "
                else:
                    query += "WHERE "
                    first = False
                query += "{0}.{1} = {2} ".format(
                        alias, group_by_col,
                        instance_id)

            # negation filters
            if len(negations) > 0:
                for neg_pred in negations:

                    if not first:
                        query += "AND "
                    else:
                        query += "WHERE "
                        first = False

                    neg_variables = [var.arg for var in neg_pred.variables]
                    neg_vars = set(pred_variables) & set(neg_variables)
                    for var in neg_vars:
                        index_pred = pred_variables.index(var)
                        query += "{0}.{1} NOT IN {2} ".format(
                                pred.name,
                                self.table_index[pred.name][index_pred],
                                neg_pred.name
                                )
            if query in executed_queries:
                continue

            executed_queries.add(query)
            self.cur.execute(query)
            selected = self.cur.fetchall()

            for s in selected:
                arguments = s[:len(pred_variables)]
                head_grounding = {'name': pred.name, 'arguments': arguments, 'isneg': pred.isneg, 'obs': pred.isobs,
                                  'target_pos': pred.label_pos()}
                str_ = RuleGrounding.pred_str(head_grounding)
                if str_ not in gold_data:
                    gold_data.add(str_)
                    ret.append(head_grounding)
        return ret

    def get_target_counts(self, rule_template, group_by,
                          instance_id, filter_by=[]):

        predicate_templates = rule_template.body + [rule_template.head]
        join_vars = set([var for (var, _, _) in filter_by + rule_template.filters])
        target_counts = self.select_target_counts(rule_template.body, rule_template.target,
                                                  join_vars, filter_by + rule_template.filters,
                                                  group_by, instance_id)
        return dict(target_counts)


    def unfold_rule_groundings(self, rule_template, ruleidx,
                                group_by, instance_id, split_on_value,
                                head_ground_truth, isconstr=False,
                                filter_by=[]):

        predicate_templates = rule_template.body + [rule_template.head]
        target_vars = set(rule_template.target_vars())

        target_preds = []
        for pred in predicate_templates:
            if not pred.isobs:
                target_preds.append(True)
            else:
                appended = False
                for var in pred.variables:
                    #print var, var.isobs
                    if not var.isobs:
                        target_preds.append(True)
                        appended = True
                        break
                if not appended:
                    target_preds.append(False)

        target_const = set(rule_template.target_constants())

        if isconstr and (set(rule_template.head.variables) & target_vars) == 0:
            target_preds[-1] = True

        forced_tpos = None
        if rule_template.split_on is not None:
            forced_tpos = rule_template.head.var_pos[rule_template.split_on]
            forced_ttype = rule_template.split_on_ttype

        # Take care of the issue of unknowns in the predicate coming from seeding
        body_ranges, body_targets, head_range, head_target, target_abs_pos,\
        known_variables, unknown_variables = \
                self.process_rule_template(rule_template)

        join_vars = set([var for (var, _, _) in filter_by + rule_template.filters])

        #print "selecting..."
        selected = self.select_grounding(predicate_templates, target_vars, target_preds,
                                         join_vars, filter_by + rule_template.filters,
                                         group_by, instance_id, isconstr,
                                         rule_template.split_on, split_on_value,
                                         known_variables, unknown_variables)

        #print "selected", len(selected)
        #exit()
        ret = []
        offsets = {}

        ranges = body_ranges + [head_range]
        offset = max([r[1] for (i,r) in enumerate(ranges) if r is not None and not target_preds[i]])

        for i, pred in enumerate(predicate_templates):
            for var in pred.variables:
                if var.arg in unknown_variables and var.arg not in offsets:
                    offsets[var.arg] = offset
                    offset += 1

        # iterate over all selected rows in the database
        for s in selected:
            var_mapping = {}

            # build grounding body
            body_groundings = []
            #print rule_template

            isgroundtruth = True

            for i, pred in enumerate(rule_template.body):

                # WARNING I am assuming the variable in unknown pred either was defined before or will be the next
                # element in the selected tuple!
                if not pred.isobs:
                    arguments = []
                    for var in pred.variables:
                        if var.isconstant:
                            arguments.append(var.arg)
                        elif var.arg in var_mapping:
                            arguments.append(var_mapping[var.arg])
                        elif var.arg in offsets:
                            arguments.append(s[offsets[var.arg]])
                    obs = False
                # I am assuming the variable in neg pred was defined before!
                elif pred.isneg:
                    arguments = []
                    for var in pred.variables:
                        if var.isconstant:
                            arguments.append(var.arg)
                        elif var.arg in var_mapping:
                            arguments.append(var_mapping[var.arg])
                        elif var.arg in offsets:
                            arguments.append(s[offsets[var.arg]])
                    obs = sum(not v.isobs for v in pred.variables) == 0
                else:
                    l, r = body_ranges[i]
                    arguments = list(s[l:r])

                    # replace constants
                    for j, var in enumerate(pred.variables):
                        if var.isconstant:
                            arguments[j] = var.arg

                    for arg, var in zip(arguments, pred.variables):
                        var_mapping[var.arg] = arg
                    obs = sum(not v.isobs for v in pred.variables) == 0

                str_rep = pred.name + "(" + ",".join(map(str, arguments)) + ")"
                if target_preds[i] and str_rep not in head_ground_truth:
                    isgroundtruth = False

                #print pred.name, "arguments", arguments
                ttype = None; tpos = None

                if i in body_targets:
                    ttype, tpos, isconst, const_value = body_targets[i]

                if forced_tpos is not None and pred.name == rule_template.head.name:
                    tpos = forced_tpos
                    ttype = forced_ttype

                bg = {'name': pred.name, 'arguments': arguments, 'ttype': ttype,
                      'obs': obs, 'isneg': pred.isneg, 'target_pos': tpos}
                body_groundings.append(bg)

            # build grounding head
            pred = rule_template.head

            if head_range is not None and not isconstr and not target_preds[-1]:
                l, r = head_range
                arguments = list(s[l:r])

                for i, var in enumerate(pred.variables):
                    if var.isconstant:
                        arguments[i] = var.arg
            else:
                # if head predicate was not observed no args are retrieved
                arguments = []
                for var in pred.variables:
                    if not var.isconstant and var.arg in var_mapping:
                        arguments.append(var_mapping[var.arg])
                    elif not var.isconstant and var.arg in offsets:
                        arguments.append(s[offsets[var.arg]])
                    elif var.isconstant:
                        arguments.append(var.arg)
            str_rep = pred.name + "(" + ",".join(map(str, arguments)) + ")"
            if str_rep not in head_ground_truth:
                isgroundtruth = False


            ttype = None; tpos = None
            if head_target is not None:
                ttype, tpos, isconst, const_value = head_target

            if forced_tpos is not None:
                tpos = forced_tpos
                ttype = forced_ttype

            head_grounding = {'name': pred.name, 'arguments': arguments, 'ttype': ttype,
                              'obs': False, 'isneg': pred.isneg, 'target_pos': tpos}

            # create rule grounding and append to result
            #print var_mapping
            #print rule_template.target
            if rule_template.target is not None:
                pred_target = var_mapping[rule_template.target]
            else:
                pred_target = None

            rule_grounding = RuleGrounding(body_groundings, head_grounding,
                                           ruleidx, not rule_template.head.isobs,
                                           pred_target, isgroundtruth=isgroundtruth)
            rule_grounding.build_predicate_index()
            rule_grounding.build_body_predicate_dic()
            ret.append(rule_grounding)
        printed = 0
        '''
        for r in ret:
            if r.isgroundtruth:
                print "\tR:", r
                printed += 1
            if printed >= 20:
                break
        '''
        return ret

    def get_distinct_values(self, predicate_template, variable):
        var_pos = predicate_template.var_pos[variable]
        column = self.table_index[predicate_template.name][var_pos]
        query = "SELECT DISTINCT {0} FROM {1}".format(
                column, predicate_template.name)
        #print query
        self.cur.execute(query)
        return self.cur.fetchall()

    def list_predicates(self):
        query = "SELECT name FROM sqlite_master WHERE type='table'"
        self.cur.execute(query)
        return self.cur.fetchall()

    def list_predicate_groundings(self, table_name):
        query = "SELECT * FROM {0}"
        query = query.format(table_name)
        self.cur.execute(query)
        return self.cur.fetchall()

    def get_largest_sentence(self):
        query = "SELECT sentenceId, count(*) FROM InSentence GROUP BY sentenceId ORDER BY COUNT(*) DESC LIMIT 1"
        self.cur.execute(query)
        return self.cur.fetchone()[0]
