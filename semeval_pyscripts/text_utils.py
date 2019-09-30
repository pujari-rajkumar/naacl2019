import spacy
import nltk
import pickle
import math
import numpy as np
import jsonrpc
from simplejson import loads
from nltk.corpus import wordnet as wn
from gensim.models import KeyedVectors



spacy_nlp = spacy.load('en')

word_vectors = KeyedVectors.load_word2vec_format('/homes/rpujari/scratch/word2vec/w2v.bin', binary = True)
with open('/homes/rpujari/scratch/data/semeval-t11/unk_veck.pkl', 'rb') as infile:
    unk_vec = pickle.load(infile)
    
with open('/homes/rpujari/scratch/data/semeval-t11/ent_vecs/reln_list.pkl', 'rb') as infile:
    amr_reln_list = pickle.load(infile)
    
with open('/homes/rpujari/scratch/data/semeval-t11/ent_vecs/ent_vecs.pkl', 'rb') as infile:
    reln_vecs = pickle.load(infile)


class corenlp(object):
    def __init__(self):
        self.server = jsonrpc.ServerProxy(jsonrpc.JsonRpc20(), jsonrpc.TransportTcpIp(addr=("128.10.4.66", 4056)))
    def parse(self, sent):
        return loads(self.server.parse(sent))

stanford_nlp = corenlp()

def cos_similarity(v1, v2):
    ans = 0
    i = 0
    while i < len(v1):
        if not (math.isnan(v1[i]) or math.isnan(v2[i])):
            ans = ans + (v1[i] * v2[i])
        else:
            return 0.0
        i = i + 1
    if np.linalg.norm(v1) != 0 and np.linalg.norm(v2) != 0:
        ret = (ans / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    else:
        return 0.0
    if math.isnan(ret):
        return 0.0
    return ret


def get_wv(w1):
    try:
        return word_vectors[w1]
    except:
        return unk_vec


def get_wv_similarity(w1, w2):
    return cos_similarity(get_wv(w1), get_wv(w2))


def get_wn_similarity(w1, w2):
    allsyns1 = wn.synsets(w1)
    allsyns2 = wn.synsets(w2)
    try:
        #best = max((wn.wup_similarity(s1, s2) or 0, s1, s2) for s1, s2 in product(allsyns1, allsyns2))
        best = [wn.wup_similarity(allsyns1[0], allsyns2[0])]
        if best[0]:
            return best[0]
        else:
            return 0
    except:
        return 0


