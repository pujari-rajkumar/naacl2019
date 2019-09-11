class FeatureFunction(object):

    def __init__(self, feature_name, feature_type,
                 embedding_container=None, vocab_size=None,
                 embedding_size=None, vocab_index=None):
        self.feat_name = feature_name
        self.feat_type = feature_type
        self.embedding_container = embedding_container
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.vocab_index = vocab_index

    def __str__(self):
        """
        Obtain a string representation of a feat func of the form:
        type(name, **args)

        Returns:
            A string representation of the function
        """
        if self.feat_type == FeatureType.Embedding:
            ret = "embedding({0}, {1}, {2}, {3}, {4})".format(
                    self.feat_name, self.embedding_container,
                    self.vocab_size, self.embedding_size,
                    self.vocab_index)
        elif self.feat_type == FeatureType.Vector:
            ret = "vector({0})".format(self.feat_name)
        elif self.feat_type == FeatureType.Input:
            ret = "input({0})".format(self.feat_name)
        return ret

    def __repr__(self):
        """
        Use the string representation for feature function when
        printing collections of features

        Returns:
            A string representation of the feat function
        """
        return str(self)

class FeatureType(object):
    """
    Enum type for a feature type
    Vector: will be concatenated with other vectors at preproc time
    Input: will be passed in a list to the neural architectures
    """
    Embedding, Vector, Input = range(0, 3)
