import torch
import numpy as np
import h5py

class snli_data(object):
    '''
        class to handle training data
    '''

    def __init__(self, fname, max_length):

        if max_length < 0:
            max_length = 9999

        f = h5py.File(fname, 'r')
        self.source = torch.from_numpy(np.array(f['source'])) - 1
        self.target = torch.from_numpy(np.array(f['target'])) - 1
        self.src_question = torch.from_numpy(np.array(f['src_questions'])) - 1
        self.targ_question = torch.from_numpy(np.array(f['targ_questions'])) - 1
        self.have_ques = int(np.array(f['have_ques']))
        self.label = torch.from_numpy(np.array(f['label'])) - 1
        self.label_size = torch.from_numpy(np.array(f['label_size']))
        self.source_l = torch.from_numpy(np.array(f['source_l']))
        self.target_l = torch.from_numpy(np.array(f['target_l'])) # max target length each batch
        self.src_ques_l = torch.from_numpy(np.array(f['src_ques_l']))
        self.targ_ques_l = torch.from_numpy(np.array(f['targ_ques_l']))
        # idx in torch style; indicate the start index of each batch (starting
        # with 1)
        self.src_ids = torch.from_numpy(np.array(f['src_ids']))
        self.targ_ids = torch.from_numpy(np.array(f['targ_ids']))
        self.batch_idx = torch.from_numpy(np.array(f['batch_idx'])) - 1
        self.batch_l = torch.from_numpy(np.array(f['batch_l']))

        self.batches = []   # batches
        self.id_batches = []

        self.length = self.batch_l.size(0)  # number of batches

        self.size = 0   # number of sentences

        for i in range(self.length):
            if self.source_l[i] <= max_length and self.target_l[i] <= max_length:
                if i != 0  and self.batch_idx[i] != self.batch_idx[i - 1] + self.batch_l[i - 1]:
                    print str(self.batch_idx[i]) + '    ' + str(self.batch_idx[i] + self.batch_l[i])
                if i == self.length - 1:
                    batch = (self.source[self.batch_idx[i] : ][:, :self.source_l[i]],
                           self.target[self.batch_idx[i] : ][:, :self.target_l[i]],
                           self.src_question[self.batch_idx[i] : ][:, :self.src_ques_l[i]],
                           self.targ_question[self.batch_idx[i] : ][:, :self.targ_ques_l[i]],
                           self.label[self.batch_idx[i] : ],
                          )
                    id_batch = (self.src_ids[self.batch_idx[i] : ],
                           self.targ_ids[self.batch_idx[i] : ],
                          )        
                else:
                    batch = (self.source[self.batch_idx[i] : self.batch_idx[i] + self.batch_l[i]][:, :self.source_l[i]],
                           self.target[self.batch_idx[i] : self.batch_idx[i] + self.batch_l[i]][:, :self.target_l[i]],
                           self.src_question[self.batch_idx[i] : self.batch_idx[i] + self.batch_l[i]][:, :self.src_ques_l[i]],
                           self.targ_question[self.batch_idx[i] : self.batch_idx[i] + self.batch_l[i]][:, :self.targ_ques_l[i]],
                           self.label[self.batch_idx[i] : self.batch_idx[i] + self.batch_l[i]],
                          )
                    id_batch = (self.src_ids[self.batch_idx[i] : self.batch_idx[i] + self.batch_l[i]],
                           self.targ_ids[self.batch_idx[i] : self.batch_idx[i] + self.batch_l[i]],
                           )

                assert(batch[0].size()[0] == id_batch[0].size()[0])
                self.id_batches.append(id_batch)
                self.batches.append(batch)
                self.size += self.batch_l[i]

class w2v(object):

    def __init__(self, fname):
        f = h5py.File(fname, 'r')
        self.word_vecs = torch.from_numpy(np.array(f['word_vecs']))

