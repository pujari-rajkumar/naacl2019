import numpy as np
import os
from sklearn.metrics import *
from drail.learn.local_learner import LocalLearner
from drail.learn.global_learner import GlobalLearner

def main():
    path = 'examples/multirc/'
    rule_file = os.path.join(path, 'rule.dr')
    np.random.seed(147)

    learner = LocalLearner()
    learner.compile_rules(rule_file)
    db = learner.create_dataset(path)

    # Since we are not using neural nets, we are not extracting the binary classifiers
    # Hot fix: add them manually
    learner.binary_classifiers.add("Contradict")
    learner.binary_classifiers.add("Entail")

    learner.build_feature_extractors(
        db, prob_score_fp = os.path.join(path, 'test_probs.json'), ent_fp = os.path.join(path, 'test_ent_probs.json'), contr_fp = os.path.join(path, 'test_contr_probs.json'),
        femodule_path=path)

    learner.extract_data(db)
    res, preds = learner.predict(db, fold='test', scmodule_path=path, get_predicates=True)

    print(preds)

if __name__ == "__main__":
    main()

