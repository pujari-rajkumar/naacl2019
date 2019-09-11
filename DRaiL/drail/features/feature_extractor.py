import numpy as np
from ..model.feature_function import FeatureType

class FeatureExtractor(object):

    def __init__(self):
        '''
        an extractor is defined by training rules observed
        '''
        self.rng = np.random.RandomState(123)
        self.notification = True

    def build(self):
        '''
        record observations from the training rules
        should be defined in child extractor
        if nothing needs to be recorded it will pass
        '''
        # i removed the nonimplemented error because we want
        # this method to just do nothing when called if not
        # overriden
        return

    def extract(self, rule_grd, functions):
        '''
        extract features for a single unobserved or observed rule
        receive a list of feature function names and call them
        using reflection
        '''
        ret = {'vector': [], 'embedding': {}, 'out': [], 'input': []}
        ret_vector = []

        # report features being extracted
        '''
        print 'Features being used'
        for func in functions:
            print func
        '''
        # stack vectors and accumulate embedding
        for func in functions:
            temp = getattr(self, func.feat_name)(rule_grd)

            if func.feat_type == FeatureType.Vector:
                ret_vector.append(temp)
            elif func.feat_type == FeatureType.Embedding:
                ret['embedding'][func.feat_name] = temp
            elif func.feat_type == FeatureType.Input:
                ret['input'].append(temp)
        if len(ret_vector) > 0:
            ret['vector'] = list(np.hstack(ret_vector))
        return ret

    def sequential_out_index(self, rule_grd):
        '''
        this method must be overriden by featture classes
        for problems that have sequential output
        '''
        raise NotImplementedError

    def extract_multiclass_head(self, rule_grd):
        '''
        this method must be overriden by feature classes
        for problems that have multiclass rules
        '''
        raise NotImplementedError

    def get_class(self, label):
        '''
        this method must be overriden by feature classes
        for problems that have multiclass rules
        '''
        return NotImplementedError

    def get_label(self, label):
        '''
        this method must be overriden by feature classes
        for problems that have multiclass rules
        '''
        return NotImplementedError
