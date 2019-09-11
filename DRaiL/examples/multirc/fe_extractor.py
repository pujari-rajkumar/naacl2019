import json
import numpy as np

from drail.features.feature_extractor import FeatureExtractor
from drail.features import utils

class FeExtractor(FeatureExtractor):

	def __init__(self, prob_score_fp, ent_fp, contr_fp):
		super(FeExtractor, self).__init__()
		self.prob_score_fp = prob_score_fp
		self.ent_fp = ent_fp
		self.contr_fp = contr_fp
		np.random.seed(18197)

	def build(self):
		with open(self.prob_score_fp, 'r') as f:
			self.prob_dict = json.load(f)
		with open(self.ent_fp, 'r') as f:
			self.entail_dict = json.load(f)
		with open(self.contr_fp, 'r') as f:
			self.contradict_dict = json.load(f)

	def get_cscore(self, rule_grd):
		cid = rule_grd.get_body_predicates("InQuestion")[0]['arguments'][1]
		return self.prob_dict[unicode(cid)]

	def get_entailment_score(self, rule_grd):
		cid1 = rule_grd.get_body_predicates("InQuestion")[0]['arguments'][1]
		cid2 = rule_grd.get_body_predicates("InQuestion")[1]['arguments'][1]
		try:
			return self.entail_dict[unicode(cid1)][unicode(cid2)]
		except KeyError:
			return 0.0
	
	def get_contradiction_score(self, rule_grd):
		cid1 = rule_grd.get_body_predicates("InQuestion")[0]['arguments'][1]
		cid2 = rule_grd.get_body_predicates("InQuestion")[1]['arguments'][1]
		try:
			return self.contradict_dict[unicode(cid1)][unicode(cid2)]
		except KeyError:
			return 0.0

	def extract_multiclass_head(self, rule_grd):
        	ret = rule_grd.get_head_predicate()
        	return ret['arguments'][1]

