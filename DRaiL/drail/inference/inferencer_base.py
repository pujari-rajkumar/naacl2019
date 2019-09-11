class InferencerBase(object):

    def __init__(self, ruleGroundings, constraintGroundings):
        self.ruleGroundings = ruleGroundings
        self.constraintGroundings = constraintGroundings

    def encode(self):
        raise NotImplementedError()

    def optimize(self):
        raise NotImplementedError()

    def evaluate(self, gold_heads, binary_predicates=[]):
        raise NotImplementedError()
