class EquationElement(object):

    def __init__(self, element, is_forall, forall_var):
        # element can be a predicate template or an integer
        self.element = element
        self.is_forall = is_forall
        self.forall_var = forall_var

    def __eq__(self, other):
        return (self.element == other.element and \
               self.is_forall == other.is_forall and \
               self.forall_var == forall_var)

    def __neq__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        ret = ""
        if self.is_forall:
            ret += "\sum_" + self.forall_var + "(" + str(self.element) + ")"
        else:
            ret += str(self.element)
        return ret

    def __repr__(self):
        return str(self)

class EquationTemplate(object):

    def __init__(self, left_elements, right_elements, operator):
        self.left = left_elements
        self.right = right_elements
        self.operator = operator
        self.isconstr = True

    def __eq__(self, other):
        if len(self.left) != len(other.left) or \
           len(self.right) != len(other.right) or \
           self.operator != other.operator:
               return False
        for p, p_prime in zip(self.left, other.left):
            if p != p_prime:
                return False
        for p, p_prime in zip(self.right, other.right):
            if p != p_prime:
                return False
        return True

    def __neq__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        ret = ""
        if len(self.left) > 0:
            ret += str(self.left[0])
            for i in range(1, len(self.left)):
                ret += " + " + str(self.left[i])
        ret += " " + self.operator + " "
        if len(self.right) > 0:
            ret += str(self.right[0])
            for i in range(1, len(self.right)):
                ret += " + " + str(self.right[i])
        return ret

    def __repr__(self):
        return str(self)

class EquationGrounding(object):

    ## at this stage I assume that sums have been expanded
    ## to their actual lists
    def __init__(self, left_elements, right_elements, operator,
                 equation_template, isgroundtruth=False):
        self.left = left_elements
        self.right = right_elements
        self.operator = operator
        self.equation_template = equation_template
        self.isgroundtruth = isgroundtruth

    ## TODO: check if we need the indexing methods that we use
    ## in rule templates

    def __eq__(self, other):
        if len(self.left) != len(other.left) or \
           len(self.right) != len(other.right):
               return False
        for p, p_prime in zip(self.left, other.left):
            if p != p_prime:
                return False
        for p, p_prime in zip(self.right, other.right):
            if p != p_prime:
                return False
        return True

    def __neq__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        ret = ""
        if len(self.left) > 0:
            ret += str(self.left[0])
            for i in range(1, len(self.left)):
                ret += " + " + str(self.left[i])
        ret += " " + self.operator + " "
        if len(self.right) > 0:
            ret += str(self.right[0])
            for i in range(1, len(self.right)):
                ret += " + " + str(self.right[i])
        return ret

    def __repr__(self):
        return str(self)
