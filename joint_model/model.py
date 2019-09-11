import logging
import copy
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np

from utils import vocab
from doc import batchify
from joint_qa import JointQA

logger = logging.getLogger()

class Model:

    def __init__(self, args):
        self.args = args
        self.batch_size = args.batch_size
        self.finetune_topk = args.finetune_topk
        self.lr = args.lr
        self.use_cuda = (args.use_cuda == True) and torch.cuda.is_available()
        print('Use cuda:', self.use_cuda)
        if self.use_cuda:
            torch.cuda.set_device(int(args.gpu))
        self.network = JointQA(args)
        self.init_optimizer()
        self.nli_loss_criterion = torch.nn.NLLLoss()
        if args.pretrained:
            print('Load pretrained model from %s...' % args.pretrained)
            self.load(args.pretrained)
        else:
            self.load_embeddings(vocab.tokens(), args.embedding_file)
        self.network.register_buffer('fixed_embedding', self.network.embedding.weight.data[self.finetune_topk:].clone())
        if self.use_cuda:
            self.network.cuda()
        print(self.network)
        self._report_num_trainable_parameters()


    def _report_num_trainable_parameters(self):
        num_parameters = 0
        for p in self.network.parameters():
            if p.requires_grad:
                sz = list(p.size())
                if sz[0] == len(vocab):
                    sz[0] = self.finetune_topk
                num_parameters += np.prod(sz)
        print('Number of parameters: ', num_parameters)

    def train(self, train_data):
        self.network.train()
        self.updates = 0
        iter_cnt, num_iter = 0, (len(train_data[1]) + self.batch_size - 1) // self.batch_size
        for batch_input in self._iter_data(train_data):
            feed_input = [x for x in batch_input[:-3]]
            y1 = batch_input[-3]
            y2 = batch_input[-2]
            nli_y = batch_input[-1]
            c1_proba, c2_proba, nli_proba = self.network(*feed_input)

            loss = F.binary_cross_entropy(c1_proba, y1) + F.binary_cross_entropy(c2_proba, y2) + self.nli_loss_criterion(nli_proba, nli_y)
            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.args.grad_clipping)

            # Update parameters
            self.optimizer.step()
            self.network.embedding.weight.data[self.finetune_topk:] = self.network.fixed_embedding
            self.updates += 1
            iter_cnt += 1

            if self.updates % 100 == 0:
                print('Iter: %d/%d, Loss: %f' % (iter_cnt, num_iter, loss.data.cpu().numpy()))
        self.scheduler.step()
        print('LR:', self.scheduler.get_lr()[0])

    def evaluate(self, dev_data, debug=False, eval_multirc=False):
        if len(dev_data) == 0:
            return -1.0
        self.network.eval()
        correct, total, prediction, gold = 0, 0, [], []
        #dev_data = sorted(dev_data, key=lambda ex: ex.id)
        #dev_ids = [ex.id for ex in dev_data]
        for batch_input in self._iter_data(dev_data):
            feed_input = [x for x in batch_input[:-3]]
            y1 = batch_input[-3].data.cpu().numpy()
            y2 = batch_input[-2].data.cpu().numpy()
            nli_y = batch_input[-1].data.cpu().numpy()
            c1_proba, c2_proba, nli_proba = self.network(*feed_input)
            c1_proba, c2_proba, nli_proba = c1_proba.data.cpu(), c2_proba.data.cpu(), nli_proba.data.cpu()
            prediction += [(p1, p2, p3) for p1, p2, p3 in zip(list(c1_proba), list(c2_proba), list(nli_proba))]
            gold += [(int(l1), int(l2), int(l3)) for l1, l2, l3 in zip(list(y1), list(y2), list(nli_y))]
            assert(len(prediction) == len(gold))
        
        cur_pred, cur_gold, cur_choices = {}, {}, {}

        if eval_multirc:
            #MultiRc Evaluation
            ques_pred, ques_gold = {}, {}
            if debug:
                res_writer = open('/homes/rpujari/scratch/joint_model/results/res', 'w')
                nli_writer = open('/homes/rpujari/scratch/joint_model/results/nli', 'w')
                gold_writer = open('/homes/rpujari/scratch/joint_model/results/gold', 'w')
            for i, ex in enumerate(dev_data[1]):
                c1_id, c2_id, nli_y = ex
                ex1 = dev_data[0][c1_id]
                ex2 = dev_data[0][c2_id]

                cur_pred[c1_id] = prediction[i][0]
                cur_pred[c2_id] = prediction[i][1]
                cur_gold[c1_id] = gold[i][0]
                cur_gold[c2_id] = gold[i][1]
                cur_choices[c1_id] = dev_data[0][c1_id].choice
                cur_choices[c2_id] = dev_data[0][c2_id].choice

                q1_id = c1_id[:-2]
                q2_id = c2_id[:-2]

                if q1_id not in ques_pred:
                    ques_pred[q1_id] = []
                    ques_gold[q1_id] = []

                if q2_id not in ques_pred:
                    ques_pred[q2_id] = []
                    ques_gold[q2_id] = []
                
                if prediction[i][0] >= 0.5:
                    ques_pred[q1_id].append(1)
                else:
                    ques_pred[q1_id].append(0)
                if prediction[i][1] >= 0.5:
                    ques_pred[q2_id].append(1)
                else:
                    ques_pred[q2_id].append(0)
                
                ques_gold[q1_id].append(gold[i][0])
                ques_gold[q2_id].append(gold[i][1])
              
                nli_pred = F.softmax(prediction[i][2], dim=0).data.cpu().numpy()
                nli_pred_y = np.argmax(nli_pred)

                if debug:
                    nli_writer.write(c1_id + '\t' + c2_id + '\t' + str(nli_pred_y) + '\t' + str(nli_pred[nli_pred_y]) + '\n')

            for c_id in cur_pred:
                if cur_pred[c_id].data.cpu().numpy() >= 0.5:
                    py = 1
                else:
                    py = 0
                gy = cur_gold[c_id]
                if debug:
                    res_writer.write(c_id + '\t' + str(cur_pred[c_id].data.cpu().numpy()) + '\n')
                    gold_writer.write(c_id + '\t' + str(gy) + '\n')
                if py == gy:
                    correct += 1
                total += 1

            corr_ques = 0.
            tot_ques = 0.

            for q_id in ques_pred:
                q_corr = True
                for py, gy in zip(ques_pred[q_id], ques_gold[q_id]):
                    if py != gy:
                        q_corr = False
                        break
                if q_corr:
                    corr_ques += 1
                tot_ques += 1

            acc = 1.0 * correct / total
            #acc = 1.0 * corr_ques / tot_ques

            if debug:
                res_writer.close()
                nli_writer.close()
                gold_writer.close()
            return acc
	
        #Other datasets
        if debug:
            res_writer = open('/homes/rpujari/scratch/joint_model/results/res', 'w')
            nli_writer = open('/homes/rpujari/scratch/joint_model/results/nli', 'w')
            gold_writer = open('/homes/rpujari/scratch/joint_model/results/gold', 'w')
        for i, ex in enumerate(dev_data[1]):
            c1_id, c2_id, nli_y = ex
            ex1 = dev_data[0][c1_id]
            ex2 = dev_data[0][c2_id]
            
            cur_pred[c1_id] = prediction[i][0]
            cur_pred[c2_id] = prediction[i][1]
            cur_gold[c1_id] = gold[i][0]
            cur_gold[c2_id] = gold[i][1]
            cur_choices[c1_id] = dev_data[0][c1_id].choice
            cur_choices[c2_id] = dev_data[0][c2_id].choice

            nli_pred = F.softmax(prediction[i][2], dim=0).data.cpu().numpy()
            nli_pred_y = np.argmax(nli_pred)

            if debug:
                nli_writer.write(c1_id + '\t' + c2_id + '\t' + str(nli_pred_y) + '\t' + str(nli_pred[nli_pred_y]) + '\n')
         
        for c_id in cur_pred:
            if c_id.endswith('0'):
                py = np.argmax([cur_pred[c_id], cur_pred[c_id[:-1] + '1']])
                gy = np.argmax([cur_gold[c_id], cur_gold[c_id[:-1] + '1']])
                if debug:
                    res_writer.write(c_id[:-1] + '1' + '\t' + str(py) + '\t' + str(cur_pred[c_id[:-1] + '1'].data.cpu().numpy()) + '\n')
                    res_writer.write(c_id + '\t' + str(abs(py - 1)) + '\t' + str(cur_pred[c_id].data.cpu().numpy()) + '\n')
                    gold_writer.write(c_id[:-1] + '1' + '\t' +  str(gy) + '\n')
                    gold_writer.write(c_id + '\t' + str(abs(gy - 1)) + '\n')
                if py == gy:
                    correct += 1
                total += 1

        acc = 1.0 * correct / total
        if debug:
            res_writer.close()
            nli_writer.close()
            gold_writer.close()
        return acc

    def predict(self, test_data):
        # DO NOT SHUFFLE test_data
        self.network.eval()
        prediction = []
        for batch_input in self._iter_data(test_data):
            feed_input = [x for x in batch_input[:-3]]
            pred_proba = self.network(*feed_input)
            pred_proba = pred_proba.data.cpu()
            prediction += list(pred_proba)
        return prediction

    def _iter_data(self, data):
        choice_data, nli_data = data
        num_iter = (len(nli_data) + self.batch_size - 1) // self.batch_size
        for i in range(num_iter):
            start_idx = i * self.batch_size
            batch_data = nli_data[start_idx:(start_idx + self.batch_size)]
            batch_input = batchify(choice_data, batch_data)

            # Transfer to GPU
            if self.use_cuda:
                batch_input = [Variable(x.cuda(async=True)) for x in batch_input]
            else:
                batch_input = [Variable(x) for x in batch_input]
            yield batch_input

    def load_embeddings(self, words, embedding_file):
        """Load pretrained embeddings for a given list of words, if they exist.

        Args:
            words: iterable of tokens. Only those that are indexed in the
              dictionary are kept.
            embedding_file: path to text file of embeddings, space separated.
        """
        words = {w for w in words if w in vocab}
        logger.info('Loading pre-trained embeddings for %d words from %s' %
                    (len(words), embedding_file))
        embedding = self.network.embedding.weight.data

        # When normalized, some words are duplicated. (Average the embeddings).
        vec_counts = {}
        with open(embedding_file) as f:
            for line in f:
                parsed = line.rstrip().split(' ')
                assert(len(parsed) == embedding.size(1) + 1)
                w = vocab.normalize(parsed[0])
                if w in words:
                    vec = torch.Tensor([float(i) for i in parsed[1:]])
                    if w not in vec_counts:
                        vec_counts[w] = 1
                        embedding[vocab[w]].copy_(vec)
                    else:
                        logging.warning('WARN: Duplicate embedding found for %s' % w)
                        vec_counts[w] = vec_counts[w] + 1
                        embedding[vocab[w]].add_(vec)

        for w, c in vec_counts.items():
            embedding[vocab[w]].div_(c)

        logger.info('Loaded %d embeddings (%.2f%%)' %
                    (len(vec_counts), 100 * len(vec_counts) / len(words)))

    def init_optimizer(self):
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(parameters, self.lr,
                                       momentum=0.4,
                                       weight_decay=0)
        elif self.args.optimizer == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                        lr=self.lr,
                                        weight_decay=0)
        else:
            raise RuntimeError('Unsupported optimizer: %s' %
                               self.args.optimizer)
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[5, 8], gamma=0.5)

    def save(self, ckt_path):
        state_dict = copy.copy(self.network.state_dict())
        if 'fixed_embedding' in state_dict:
            state_dict.pop('fixed_embedding')
        params = {'state_dict': state_dict}
        torch.save(params, ckt_path)

    def load(self, ckt_path):
        logger.info('Loading model %s' % ckt_path)
        saved_params = torch.load(ckt_path, map_location=lambda storage, loc: storage)
        state_dict = saved_params['state_dict']
        return self.network.load_state_dict(state_dict)

    def cuda(self):
        self.use_cuda = True
        self.network.cuda()
