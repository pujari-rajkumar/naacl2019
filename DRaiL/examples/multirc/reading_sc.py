import numpy as np

def class_prob(features, fe_class):
	prob_scores = []
	for i, cscore in enumerate(features['vector']):
		score_val = float(cscore[0])
		prob_scores.append([1.0 - score_val, score_val])
	return np.array(prob_scores)

