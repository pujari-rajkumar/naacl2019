from label import LabelType
from argument import Argument

class PredicateTemplate(object):

    def __init__(self, name, variables, isneg=False, isobs=True):
        """
        Create a predicate template and its variables. A predicate consist
        of a relation between one or two variables.

        Args:
            name: name of the predicate relation
            variables: list of variables as strings: X, Y, etc.
            neg: boolean indicating whether the predicate is negated
        """
        self.name = name
        self.variables = variables
        self.isneg = isneg
        self.isobs = isobs
        self.var_pos = {}

        for i, var in enumerate(self.variables):
            self.var_pos[var.arg] = i

        self.var_set = set(self.var_pos.keys())

    def __eq__(self, other):
        """
        Check if `other` predicate template is equal

        Args:
            other: predicate template to compare

        Returns:
            True if they are equal, False otherwise
        """
        if self.name != other.name or \
           len(self.variables) != len(other.variables) or \
           self.isneg != other.isneg:
               return False
        # now check variables one by one
        for var, var_prime in zip(self.variables, other.variables):
            if var != var_prime:
                return False
        return True

    def __neq__(self, other):
        """
        Check if `other` predicate template is not equal

        Args:
            other: predicate template to compare

        Returns:
            True if they are not equal, False otherwise
        """
        return not self.__eq__(other)

    @staticmethod
    def str2predicate(string_repr):
        lpar_index = string_repr.index('(')
        rpar_index = string_repr.index(')')
        vars_ = string_repr[lpar_index + 1:rpar_index].split(',')

        variables = []
        for var in vars_:
            if var[0] == '"' and var[-1] == '"':
                variables.append(Argument(var[1:-1], True, True, None))
            else:
                variables.append(Argument(var, False, True, None))

        if string_repr[0] == '~':
            name = string_repr[1:lpar_index]
            isneg = True
        else:
            name = string_repr[:lpar_index]
            isneg = False

        return PredicateTemplate(name, variables, isneg, True)


    def __str__(self):
        """
        Obtain a string representation of a predicate template of the form:
        p(v1, v2)

        Returns:
            A string representation of the template
        """
        temp = map(str, self.variables)
        temp = ",".join(temp)
        if self.isneg:
            ret = "~{0}({1})"
        else:
            ret = "{0}({1})"

        if not self.isobs:
            ret += "^?"
        return ret.format(self.name, temp)

    def __repr__(self):
        """
        Use the string representation for predicate templates when
        printing collections of predicates

        Returns:
            A string representation of the template
        """
        return str(self)

    # OJO: assuming there is a single unknown variable in a predicate
    def label_type(self):
        for var in self.variables:
            if not var.isobs:
                return var.label.l_type
        return None

    # OJO: assuming there is a single unknown variable in a predicate
    def label_pos(self):
        for i, var in enumerate(self.variables):
            if not var.isobs:
                return i
        return None

    def label_var(self):
        for var in self.variables:
            if not var.isobs and not var.isconstant:
                return var
        return None

    def label_const(self):
        for var in self.variables:
            if not var.isobs and var.isconstant:
                return var
        return None
