class Label(object):

    def __init__(self, name, l_type, n_classes):
        self.name = name
        self.l_type = l_type
        self.n_classes = n_classes

    def __str__(self):
        if self.l_type == LabelType.Multilabel:
            typ = "Multilabel"
        elif self.l_type == LabelType.Multiclass:
            typ = "Multiclass"
        else:
            typ = "Binary"
        return self.name + "-" + typ + "-" + self.n_classes

class LabelType(object):
    """
    Enum type for a label type
    """
    Binary, Multiclass, Multilabel = range(0, 3)
