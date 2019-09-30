import spacy
import nltk
import jsonrpc
from simplejson import loads

spacy_nlp = spacy.load('en')

class corenlp(object):
    def __init__(self):
        self.server = jsonrpc.ServerProxy(jsonrpc.JsonRpc20(), jsonrpc.TransportTcpIp(addr=("128.10.4.66", 4056)))
    def parse(self, sent):
        return loads(self.server.parse(sent))


class amr_graph(object):
    def __init__(self, text_fp, nodes, docs):
        self.text = text_fp
        self.nodes = nodes
        self.docs = docs
        if len(self.text.keys()) == 0:
            self.empty = True
        else:
            self.empty = False
        
    def convert_to_dot(self, uid=''):
        dot_str = ''
        n_keys = sorted(self.nodes.keys())
        dot_str += 'digraph amr{\n'
        i = 0
        for key in n_keys: 
            og = [(x[0], x[1]) for x in self.nodes[key].outgoing]
            for k1, k2 in og:
                k1_idx = n_keys.index(k1)
                c1 = self.nodes[key].word
                if c1 == '-':
                    c1 = 'neg'
                c1 = (c1  + '_' + str(i) + '_' + uid).replace('-', '_')
                c2 = self.nodes[k1].word
                if c2 == '-':
                    c2 = 'neg'
                c2 = (c2 + '_' + str(k1_idx) + '_' + uid).replace('-', '_')
                dot_str += '\t' + c1 + ' -> ' + c2 + ' [label=' + k2 + '];\n'
            i += 1
        dot_str += '}\n'
        return dot_str


class amr_node(object):
    def __init__(self, word, pos, n_type, sent_id, span_id, n_id):
        self.word = word
        self.pos = pos
        self.outgoing = []
        self.sent_id = sent_id
        self.span_id = span_id
        self.n_id = n_id
        self.n_type = n_type
        self.polarity = 1.0
    def add_outgoing(self, n_id, e_type):
        self.outgoing.append((n_id, e_type))


def read_amr(fp, indices):
    f_ptr = open(fp, 'r')
    f_text = f_ptr.read()
    a_split = f_text.split('\n\n')
    
    raw_text = {}
    text_docs = {}
    nodes = {}
    nodes['0'] = amr_node(u'root', u'ROOT', u'root', -1, (-1, -1), u'ROOT')
    l_num = 0
    
    for seg in a_split:
        lines = seg.split('\n')
        if len(lines) > 3:
            if lines[-1].strip() == '(a / amr-empty)':
                return amr_graph(raw_text, nodes, text_docs)
            begin = 0
            end = 0
            for i in range(len(lines)):
                if begin == 0 and lines[i][0] != '#':
                    begin = i
                if begin != 0 and lines[i].strip() == '':
                    end = i - 1
            if end == 0:
                end = len(lines) - 1
        else:
            continue
          
        if l_num in indices:
            for line in lines[:begin]:
                if line.startswith('# ::tok'):
                    r_text = line.split()[2:]
                    pos_tags = nltk.pos_tag(r_text, tagset='universal')
                    t_str = ' '.join(r_text)
                    if type(t_str) == type('string'):
                        t_uni = t_str.decode('utf-8')
                    else:
                        t_uni = t_str
                    text_docs[l_num] = spacy_nlp(t_uni)
                    raw_text[l_num] = t_str
                if line.startswith('# ::alignments'):
                    align = line.split()[2:-6]
            align_dict = parse_align(align, l_num)

            parent_stack = []
            n_no = [l_num, 0]
            for line in lines[begin:end+1]:
                l_wsp, reln, word, n_type = parse_line(line)
                if type(word) == type('string'):
                    word = word.decode('utf-8', errors='ignore')
                    
                #Computing n_id, p_id
                if len(parent_stack) == 0:
                    parent_stack.append((l_wsp, l_num))
                else:
                    p_wsp = l_wsp
                    c_id = -1
                    while len(parent_stack) > 0 and parent_stack[-1][0] >= l_wsp:
                        if parent_stack[-1][0] > l_wsp:
                            n_no.pop()
                        if parent_stack[-1][0] == l_wsp:
                            c_id = n_no[-1]
                        p_wsp = parent_stack[-1][0]
                        parent_stack.pop()
                    c_id += 1
                    if c_id == 0:
                        n_no.append(c_id)
                    else:
                        n_no[-1] = c_id
                    parent_stack.append((l_wsp, l_num))

                n_id = '.'.join([str(x) for x in n_no])
                p_id = '.'.join([str(x) for x in n_no[:-1]])
                
                #Aligning with text span
                try:
                    span_id = align_dict[n_id]
                    n_pos = pos_tags[span_id[0]][1]
                    if n_pos == 'PRON':
                        n_pos = 'NOUN'
                except:
                    #print line.strip()
                    #print 'no alignment found - node ignored'
                    continue
                #Creating and adding node to node_dict 
                nodes[n_id] = amr_node(word, n_pos, n_type, l_num, span_id, n_id)
                if p_id != str(l_num):
                    rel = parse_relation(reln)
                    if rel[0] == 'polarity' and word == '-':
                        nodes[p_id].polarity = -1.0
                    elif rel[1] > 0:
                        nodes[p_id].add_outgoing(n_id, rel[0])
                    else:
                        nodes[n_id].add_outgoing(p_id, rel[0])
                else:
                    nodes['0'].add_outgoing(n_id, p_id)
        l_num += 1
    return amr_graph(raw_text, nodes, text_docs)


def parse_relation(s1):
    excep = ['consist-of', 'compared-to', 'employed-by']
    if s1 not in excep:
        split = s1.split('-')
        if len(split) == 2:
            return (split[0], -1)
        else:
            return (split[0], 1)
    else:
        return (s1, 1)

def parse_line(s1):
    l_wsp = len(s1) - len(s1.lstrip())
    s1_ = s1
    s1_r = s1_.replace("(", "")
    s1 = s1_r.replace(")", "")
    cols_ = s1.strip().split()
    reln_ = cols_[0][1:]
    n_id = ''
    n_type = 'ARG'
    for s1 in cols_[1:]:
        n_id += s1
    if '/' in n_id:
        word_ = n_id.split('/')[1]
        if word_ != '-' and '-' in word_:
            word_ = ' '.join(word_.split('-')[:-1])
            if word_ != 'have rel role':
                n_type = 'ACT'
    else:
        word_ = n_id
        if word_ != '-' and '-' in word_:
            word_ = ' '.join(word_.split('-')[:-1])
            if word_ == 'have rel role':
                n_type = 'ACT'
    
    return (l_wsp, reln_, word_, n_type)


def parse_align(l1, l_num):
    ret_dict = {}
    seed = str(l_num) + '.'
    for item in l1:
        cols = item.split('|')
        col_0 = cols[0].strip().split('-')
        col_1 = cols[1].strip().split('+')
        beg = int(col_0[0].strip())
        end = int(col_0[1].strip())
        for i1 in col_1:
            ret_dict[seed + i1] = (beg, end)
    return ret_dict


#Coreference on AMR

def run_coref(nodes, r_text):
    p_str = ''
    for key in r_text.keys():
        p_str += ' ' + r_text[key]
    p_str = p_str.strip()
    parser = corenlp()
    try:
        coref = parser.parse(p_str)['coref']
        i = 0
        for entry in coref:
            root = entry[0][1]
            r_key = find_rep_node(nodes, root[1], root[2], (root[3], root[4]))
            if r_key:
                for item in entry:
                    reps = find_all_reps(nodes, item[0][1], item[0][2], (item[0][3], item[0][4]))
                    for k1 in reps:
                        if k1 != r_key:
                            merge_node(nodes, k1, r_key)
            i += 1
    except KeyError:
        return


def merge_node(nodes, k1, k2):
    src_out = nodes[k1].outgoing
    for e1 in src_out:
        nodes[k2].add_outgoing(e1[0], e1[1])
    n_keys = nodes.keys()
    for n_key in n_keys:
        edges = nodes[n_key].outgoing
        i = 0
        mod_e = []
        for e1 in edges:
            if e1[0] == k1:
                mod_e.append(i)
                nodes[n_key].add_outgoing(k2, e1[1])
            i += 1
        for idx in reversed(mod_e):
            nodes[n_key].outgoing.pop(idx)
    del nodes[k1]



def find_rep_node(nodes, s_id, incl_idx, span):
    n_keys = nodes.keys()
    for key in n_keys:
        if key.startswith(str(s_id) + '.'):
            rep_node = nodes[key]
            r_span_id = rep_node.span_id
            if r_span_id[0] <= incl_idx and r_span_id[1] > incl_idx:
                return key
    max_match = None
    max_score = -1
    for key in n_keys:
        if key.startswith(str(s_id) + '.'):
            rep_node = nodes[key]
            r_span_id = rep_node.span_id
            if r_span_id[0] <= span[0]:
                score = r_span_id[1] - span[0]
            else:
                score = span[1] - r_span_id[0]
            if score > max_score:
                max_match = key
                max_score = score
    return max_match


def find_all_reps(nodes, s_id, incl_idx, span):
    n_keys = nodes.keys()
    reps = []
    for key in n_keys:
        if key.startswith(str(s_id) + '.'):
            rep_node = nodes[key]
            r_span_id = rep_node.span_id
            if r_span_id[0] <= incl_idx and r_span_id[1] > incl_idx:
                reps.append(key)
            elif r_span_id[0] <= span[0] and span[0] < r_span_id[1]:
                reps.append(key)
            elif span[0] <= r_span_id[0] and r_span_id[0] < span[1]:
                reps.append(key)
    return reps

