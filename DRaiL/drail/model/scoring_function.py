class ScoringFunction(object):

    def __init__(self, sctype, scref=None, scfunc=None):
        self.sctype = sctype
        self.scref = scref
        self.scfunc = scfunc

class ScoringType(object):
    """
    Enum type for a feature type
    """
    NNet, NNetRef, Func = range(0, 3)
