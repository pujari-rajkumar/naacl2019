import spacy
import nltk
import pickle
import math
import numpy as np
from nltk.corpus import wordnet as wn
from gensim.models import KeyedVectors
import amr_utils
import text_utils

class ent_act_graph(object):
    def __init__(self, doc=None, amr=None):
        self.ent_acts = []
        self.s_doc = None
        self.s_amr = None
        if doc:
            self.s_doc = doc
            self.ent_acts += ent_act_doc(self.s_doc)
        if amr:
            self.s_amr = amr
            self.ent_acts += ent_act_amr(self.s_amr)
    def best_match(self, eag2):
        best_score = -float('Inf')
        best_match = []
        for ea1 in self.ent_acts:
            for ea2 in eag2.ent_acts:
                t_score = ent_act_similarity(ea1, ea2)
                if t_score > best_score:
                    best_score = t_score
                    best_match = (ea1, ea2)
        if best_score == -float('Inf'):
            return (0, [])
        return (best_score, best_match)


def unigram_matches(d1, d2):
    lem1 = [x.lemma_ for x in d1]
    lem2 = [x.lemma_ for x in d2]
    c = 0.0
    for t1 in lem1:
        if t1 in lem2:
            c = c + 1.0
    den = max(len(lem1), len(lem2))
    if den > 0:
        c /= den
    else:
        return 0
    return c


def bigram_matches(d1, d2):
    c = 0.0
    lem1 = [x.lemma_ for x in d1]
    lem2 = [x.lemma_ for x in d2]
    l1 = zip(lem1[:-1], lem1[1:])
    l2 = zip(lem2[:-1], lem2[1:])
    for t1 in l1:
        if t1 in l2:
            c = c + 1.0
    den = max(len(l1), len(l2))
    if den > 0:
        c /= den
    else:
        return 0
    return c


def get_doc_vec(doc):
    d_vec = np.zeros(word_vectors['dog'].shape)
    vec_tot = 0
    for tok in doc:
        try:
            d_vec += word_vectors[tok.text]
            vec_tot += 1
        except:
            continue
    return (d_vec / vec_tot)

def syn_root(doc):
    for token in doc:
        if token.head == token:
            return token


def syn_root_match(doc1, doc2):
    wv1 = syn_root(doc1)
    wv2 = syn_root(doc2)

    try:
        wv_sim = text_utils.cos_similarity(word_vectors[wv1.text], word_vectors[wv2.text])
    except:
        wv_sim = 0
    
    allsyns1 = wn.synsets(syn_root(doc1).lemma_)
    allsyns2 = wn.synsets(syn_root(doc2).lemma_)
    
    try:
        #best = max((wn.wup_similarity(s1, s2) or 0, s1, s2) for s1, s2 in product(allsyns1, allsyns2))
        best = wn.wup_similarity(allsyns1[0], allsyns2[0])
        if best:
            wn_sim = best
        else:
            wn_sim = 0
    except:
        wn_sim = 0
    return (wv_sim, wn_sim)


def is_negated(doc1):
    for tok in doc1:
        if tok.dep_ == 'neg':
            return 1
    return 0


def neighbor_match(a, q, z, win=2):
    n_set = set()
    nm_count = 0.0
    a_lem = [x.lemma_ for x in a]
    z_lem = [x.lemma_ for x in z]
    q_lem = [x.lemma_ for x in q]
    for token in a_lem:
        if token in z_lem:
            i = z_lem.index(token)
            j = 1
            while j <= win and i + j < len(z_lem):
                n_set.add(z_lem[i + j])
                j = j + 1
            j = 1
            while j <= win and i - j >= 0:
                n_set.add(z_lem[i - j])
                j = j + 1
    for token in n_set:
        if token in q_lem:
            nm_count += 1
    return (nm_count / len(a_lem))


def joint_match(a, q, z):
    m_count = 0.0
    a_lem = [x.lemma_ for x in a]
    z_lem = [x.lemma_ for x in z]
    q_lem = [x.lemma_ for x in q]
    for token in z_lem:
        if token in a_lem or token in q_lem:
            m_count += 1
    return (m_count / len(z_lem))

def frac_coverage(d1, d2):
    lem1 = [x.lemma_ for x in d1]
    lem2 = [x.lemma_ for x in d2]
    c = 0.0
    for l1 in lem1:
        if l1 in lem2:
            c += 1
    return c / len(lem1)


def create_ic_dict(t1):
    ic_dict = {}
    for sent in t1:
        words = [x.lemma_ for x in sent]
        for word in words:
            try:
                ic_dict[word] = ic_dict[word] + 1
            except KeyError:
                ic_dict[word] = 1
    for k1 in ic_dict.keys():
        ic_dict[k1] = np.log(1.0 + (1.0 / ic_dict[k1]))
    return ic_dict


def sliding_window_rc(a, q, z, ic_dict):
    a_lem = [x.lemma_ for x in a]
    q_lem = [x.lemma_ for x in q]
    z_lem = [x.lemma_ for x in z]
    score = 0
    t_c = 0
    for word in a_lem:
        if word in z_lem:
            score += ic_dict[word]
            t_c += 1
    for word in q_lem:
        if word in z_lem:
            score += ic_dict[word]
            t_c += 1
    return score


def ent_act_doc(doc):
    dep_list = []
    for token in doc:
        if 'subj' in token.dep_:
            subj = token
            head = token.head
            for t1 in doc:
                if ('obj' in t1.dep_ or 'comp' in t1.dep_) and t1.head == head:
                    obj = t1
                    if token.text.lower() == u'i':
                        dep_list.append([(u'narrator', 1), (head.text, 1), (t1.text, 1)])
                    else:
                        dep_list.append([(token.text, 1), (head.text, 1), (t1.text, 1)])
                    for t2 in doc:
                        if t2.dep_ == 'neg' and t2.head == obj:
                            dep_list[-1][2] = (dep_list[-1][2][0], -1)
                        if t2.dep_ == 'neg' and t2.head == head:
                            dep_list[-1][1] = (dep_list[-1][1][0], -1)
    return dep_list


def ent_act_amr(a1):
    ent_acts = []
    ent_tok = text_utils.spacy_nlp(u'entity')
    s_pol = 1
    o_pol = 1
    for n_key in a1.nodes.keys(): 
        if int(n_key.split('.')[0]) in a1.docs.keys():
            n_doc = a1.docs[int(n_key.split('.')[0])]
            node = a1.nodes[n_key]
            if node.n_type == 'ACT':
                subj = ent_tok
                obj = ent_tok
                n_pol = node.polarity
                for edge in node.outgoing:
                    if edge[1] == 'ARG0':
                        subj = n_doc[a1.nodes[edge[0]].span_id[0]]
                        s_pol = a1.nodes[edge[0]].polarity
                    if edge[1] == 'ARG1':
                        obj = n_doc[a1.nodes[edge[0]].span_id[0]]
                        o_pol = a1.nodes[edge[0]].polarity
                if subj.text.lower() == u'i':
                    ent_acts.append([(u'narrator', s_pol), (n_doc[node.span_id[0]].text, n_pol), (obj.text, o_pol)])
                else:
                    ent_acts.append([(subj.text, s_pol), (n_doc[node.span_id[0]].text, n_pol), (obj.text, o_pol)])
    return ent_acts


def act_sim(ea1, ea2):
    c = 0.0
    t = 0
    for a1 in ea1:
        for a2 in ea2:
            try:
                t_sim = text_utils.get_wv_similarity(a1[1][0], a2[1][0])
                pol = a1[1][1] * a2[1][1]
                if pol == -1:
                    t_sim = (1 - t_sim) / 2
                c += t_sim
                t += 1.0
            except:
                continue
    if t == 0:
        return 0
    return (c / t)


def ent_sim(ea_q, ea_z):
    eq = set([x[0] for x in ea_q] + [x[2] for x in ea_q])
    ez = set([x[0] for x in ea_z] + [x[2] for x in ea_z])

    ent_sim = 0.0
    ent_c = 0
    for e1 in eq:
        for e2 in ez:
            try:
                t_sim = text_utils.get_wv_similarity(e1[0], e2[0])
                pol = e1[1] * e2[1]
                if pol == -1:
                    t_sim = (1 - t_sim) / 2
                ent_sim += t_sim
                ent_c += 1
            except:
                continue
    if ent_c > 0:
        ent_sim /= ent_c
    
    return ent_sim

def ent_act_similarity(ea1, ea2):
    ret_score = 0
    for i in range(2):
        t_sim = text_utils.get_wv_similarity(ea1[i][0], ea2[i][0])
        pol = ea1[i][1] * ea2[i][1]
        if pol == -1:
            t_sim = (1 - t_sim) / 2
        ret_score += t_sim
    return ret_score


def extract_features_2(a1, a_amr, q1, z1, z_amr, ic_dict, e_id, vecs=None):
    nm1 = neighbor_match(a1, q1, z1)
    jm1 = joint_match(a1, q1, z1)
    um1 = unigram_matches(a1, z1)
    bm1 = bigram_matches(a1, z1)
    fc1 = frac_coverage(a1, z1)
    sw = sliding_window_rc(a1, q1, z1, ic_dict)
    wv_sim, wn_sim = syn_root_match(a1, z1)
    eag1 = ent_act_graph(doc=a1, amr=a_amr)
    ezg1 = ent_act_graph(doc=z1, amr=z_amr)
    en_sim = ent_sim(eag1.ent_acts, ezg1.ent_acts)
    ac_sim = act_sim(eag1.ent_acts, ezg1.ent_acts)
    best_ea = eag1.best_match(ezg1)[0]
    asim = text_utils.cos_similarity(a1.vector, z1.vector)
    qsim = text_utils.cos_similarity(q1.vector, z1.vector)
    is_neg_a = is_negated(a1)
    is_neg_z = is_negated(z1)
    is_neg_q = is_negated(q1)
    if is_neg_a == is_neg_z:
        is_neg = 1
    else:
        is_neg = 0
    if vecs:
        p_vec = vecs[0][e_id]
        q_vec = vecs[1][e_id]
        a_vec = vecs[2][e_id]
        pq_sim = text_utils.cos_similarity(p_vec, q_vec)
        qa_sim = text_utils.cos_similarity(q_vec, a_vec)
        pa_sim = text_utils.cos_similarity(p_vec, a_vec)
        y_score = pa_sim + qa_sim
        return [nm1, jm1, um1, bm1, fc1, sw, wv_sim, wn_sim, ac_sim, en_sim, best_ea, qsim, asim, is_neg, pq_sim, qa_sim, pa_sim, y_score]
    else:
        return [nm1, jm1, um1, bm1, fc1, sw, wv_sim, wn_sim, ac_sim, en_sim, best_ea, qsim, asim, is_neg]


