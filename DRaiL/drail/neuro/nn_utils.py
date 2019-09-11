import torch
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import *
import sys
import time
from collections import Counter
import time
import random

from nn_model import ModelType

random.seed(1234)

def create_layer(definition):
    return torch.nn.Linear(definition['nin'], definition['nout'])

def compute_weights(Y, output_dim):
    counts = np.bincount(Y)
    num_missing = output_dim - len(counts)
    counts = np.append(counts, np.asarray([0]*num_missing))

    # smoothing
    counts[counts == 0] = 1
    #compute weights
    weights = np.asarray(np.max(counts)*1.0/counts, dtype = "float32")
    return weights
    #eg_weights = [weights[Y[idx]] for idx in range(len(Y))]
    #return np.asarray(eg_weights, dtype='float32')

def measure(y, y_, measurement, average, pos_label):
    if measurement == "acc":
        return accuracy_score(y, y_)
    elif measurement == "f1":
        return f1_score(y, y_, average = average, pos_label= pos_label)

def index_excerpt(inputs, targets, excerpt, incl_targets=True):
    ret = {}
    ret['vector'] = [inputs['vector'][i] for i in excerpt]
    ret['input'] = [inputs['input'][i] for i in excerpt]
    ret['embedding'] = {}
    for emb in inputs['embedding']:
        ret['embedding'][emb] = [inputs['embedding'][emb][i] for i in excerpt]

    if incl_targets and targets is not None:
        return ret, [targets[i] for i in excerpt]
    else:
        return ret

def sort_sequences(inputs, targets, sort_by, incl_targets=True):

    if sort_by[0] in ['input', 'embedding']:
        seqs = [elem[sort_by[1]] for elem in inputs[sort_by[0]]]
    elif sort_by[0] == 'vector':
        seqs = inputs['vector']
    else:
        print "Invalid sort by input"
        exit()

    seq_lengths = torch.LongTensor(map(len, seqs))
    max_seq_len = seq_lengths.max()
    seq_lengths_sorted, perm_idx = seq_lengths.sort(0, descending=True)

    if not incl_targets:
        ret = index_excerpt(inputs, targets, perm_idx, incl_targets)
        return ret, perm_idx
    else:
        ret, targets_sorted = index_excerpt(inputs, targets, perm_idx, incl_targets)
        return ret, targets_sorted, perm_idx

def unsort_sequences(output, perm_idx):
    _, original_idx = perm_idx.sort(0)
    return [output[i] for i in original_idx]

def iterate_minibatches(inputs, targets, batchsize, incl_targets=True, shuffle=False):
    n_size = get_batch_length(inputs)
    indices = np.arange(n_size)
    if shuffle:
       random.shuffle(indices)

    for start_idx in range(0, n_size, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        yield index_excerpt(inputs, targets, excerpt, incl_targets)

def train_local(model, optimizer, loss_fn, X, Y, n_batchsize, class_weights,
          measure_type='acc', avg_type='binary', pos_label=1, sort=False,
          sort_by=None):

    train_batches = 0; train_loss_sum = 0.0; train_measure_sum = 0.0
    model.train()
    for (batched_X, batched_Y) in \
         iterate_minibatches(X, Y, n_batchsize, shuffle=True):

        if sort and sort_by is not None:
            batched_X, batched_Y, _ = \
                sort_sequences(batched_X, batched_Y, sort_by)

        model.zero_grad()
        model.minibatch_size = get_batch_length(batched_X)
        y_pred = model(batched_X)
        _, train_Y_pred = torch.max(y_pred, 1)

        #train_loss = model.global_loss(batched_X, train_Y_pred.data.long(), batched_Y)

        # Compute and accumulate loss
        train_loss = model.pytorch_loss(y_pred, batched_Y, loss_fn)
        train_loss_sum += train_loss.data[0]

        train_batches += 1

        # Zero gradients, perform a backward pass, and update the weights.
        train_loss.backward()
        optimizer.step()
    #print "---"
    return train_batches, train_loss_sum, train_measure_sum

def get_batch_length(X):
    _max = max(len(X['vector']), len(X['input']))
    for emb in X['embedding']:
        _max = max(_max, len(X['embedding'][emb]))
    return _max

def predict_local(model, X, sort=False, sort_by=None):
    model.eval()
    y_pred = []
    if 'pred_size' not in model.config:
        batch_size = get_batch_length(X)
    else:
        batch_size = model.config['pred_size']
    for (batched_X) in iterate_minibatches(X, None, batch_size, incl_targets=False):

        if sort and sort_by is not None:
            batched_X, perm_idx = sort_sequences(batched_X, None, sort_by, incl_targets=False)

        model.minibatch_size = get_batch_length(batched_X)
        output = model(batched_X)
        _, y_pred_batch = torch.max(output, 1)
        y_pred_batch = y_pred_batch.data
        # if I sorted, now need to unsort
        if sort and sort_by is not None:
            y_pred_batch = unsort_sequences(y_pred_batch, perm_idx)
        else:
            y_pred_batch = list(y_pred_batch)
        y_pred += y_pred_batch
    return y_pred

def predict_local_scores(model, X):
    sort=False; sort_by=None
    if "sort_sequence" in model.config:
        sort=model.config['sort_sequence']
        sort_by=model.config["sort_by"]
    model.eval()
    output = np.asarray([])
    if 'pred_size' not in model.config:
        batch_size = get_batch_length(X)
    else:
        batch_size = model.config['pred_size']
    for (batched_X) in iterate_minibatches(X, None, batch_size, incl_targets=False):
        if sort and sort_by is not None:
            batched_X, perm_idx = sort_sequences(batched_X, None, sort_by, incl_targets=False)
        model.minibatch_size = get_batch_length(batched_X)
        output_batch = model(batched_X)
        output_batch = (output_batch.data).cpu().numpy()
        # if I sorted, now need to unsort
        if sort and sort_by is not None:
            output_batch = unsort_sequences(output_batch, perm_idx)

        if output.shape[0] == 0:
            output = output_batch
        else:
            output = np.vstack([output, output_batch.data])
    return np.asarray(output)

def backpropagate_global(model, optimizer, X_ls, y_pred_ls, y_pred_index_ls,
                        y_gold_ls, y_gold_index_ls):
    model.train()
    model.zero_grad()
    loss = model.loss(X_ls, y_pred_ls, y_pred_index_ls, y_gold_ls, y_gold_index_ls)
    loss.backward()
    optimizer.step()
    return loss.data[0]

def train_local_on_epoches(model, TrainX, TrainY, DevX, DevY, TestX, TestY):
    timestamp = time.time()
    print("Neural Networks {} Training Started..".format(model.nn_id))
    patience = model.config["patience"]
    n_batchsize = model.config["batch_size"]
    n_predsize = None

    # defaults. TO-DO add defaults for the rest of the params
    measurement = "acc"; average = "binary"; weighted_examples=False
    pos_label = 1; sort=False; sort_by=None

    if "measurement" in model.config:
        measurement = model.config["measurement"]
    if "average" in model.config:
        average = model.config["average"]
    if "weighted_examples" in model.config:
        weighted_examples = model.config["weighted_examples"]
    print_stat = measurement
    if "print_stat" in model.config:
        print_stat = model.config['print_stat']
    if "pos_label" in model.config:
        pos_label = int(model.config["pos_label"])
    if "sort_sequence" in model.config:
        sort=True; sort_by=model.config["sort_by"]

    # print n_batchsize
    best_val_measure = -float("infinity")
    done_looping = False
    patience_counter = 0

    epoch = 0
    start_time = time.time()

    if weighted_examples:
        class_weights = torch.from_numpy(compute_weights(TrainY, model.output_dim))
        if model.use_gpu:
            class_weights = class_weights.cuda()
    else:
        class_weights = None

    # THIS IS HARDCODED, NEED TO ALLOW FOR DIFFERENT OPT AND LOSSES
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    #loss_fn = None
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=float(model.config['learning_rate']))
    while (not done_looping):
    #for i in range(5):
        epoch += 1
        train_batches, train_loss_sum, train_measure_sum = \
                train_local(model, optimizer, loss_fn,
                            TrainX, TrainY, n_batchsize, class_weights,
                            measurement, average, pos_label, sort, sort_by)
        if len(DevX) == 0:
            val_measure = 0.0
        else:
            val_Y_pred = predict_local(model, DevX, sort, sort_by)
            val_measure = measure(DevY, val_Y_pred, measurement, average, pos_label)

        # report metrics
        if len(TrainX) == 0:
            train_measure = 0.0
        else:
            train_Y_pred = predict_local(model, TrainX, sort, sort_by)
            train_measure = measure(TrainY, train_Y_pred, measurement, average, pos_label)
        '''
        if TestX is not None and TestY is not None:

            if len(TestX) == 0:
                test_measure = 0.0
            else:
                test_Y_pred = predict_local(model, TestX, sort, sort_by)
                test_measure = measure(TestY, test_Y_pred, measurement, average, pos_label)
        else:
            pass
        #print "--- TRAIN SCORE", train_measure
        '''

        print "epoch", epoch, "loss", train_loss_sum / (1.0 * train_batches), "train_measure", train_measure, "val_measure", val_measure
        if val_measure > best_val_measure:
            #print "epoch", epoch, "loss", train_loss_sum / (1.0 * train_batches), "train_measure", train_measure, "val_measure", val_measure
            patience_counter = 0
            best_val_measure = val_measure
            best_epoch = epoch
            model_state = model.state_dict()
            torch.save(model_state, 'best_model_{0}.pt'.format(timestamp))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                done_looping = True

    model.load_state_dict(torch.load('best_model_{0}.pt'.format(timestamp)))

    dev_Y_pred = predict_local(model, DevX, sort, sort_by)
    train_Y_pred = predict_local(model, TrainX, sort, sort_by)
    test_Y_pred = predict_local(model, TestX, sort, sort_by)

    '''
    print "\nTraining scores"
    print classification_report(TrainY, train_Y_pred)
    print "f1 macro", f1_score(TrainY, train_Y_pred, average='macro')
    #print confusion_matrix(TrainY, train_Y_pred)
    print "\nDev scores"
    #print classification_report(DevY, dev_Y_pred)
    print "f1 macro", f1_score(DevY, dev_Y_pred, average='macro')
    #print confusion_matrix(DevY, dev_Y_pred)
    '''
    if TestX is not None and TestY is not None:
        print "\tTest scores"
        print classification_report(TestY, test_Y_pred)
        print "f1 macro", f1_score(TestY, test_Y_pred, average='macro')
        print "acc", accuracy_score(TestY, test_Y_pred)
    print("\nTraining {} Epoches took {:.3f}s".format(epoch, time.time() - start_time))
    '''
    if TestX is not None and TestY is not None:
        return [measure(TrainY, train_Y_pred, print_stat, average, pos_label),
                measure(DevY, dev_Y_pred, print_stat, average, pos_label),
                measure(TestY, test_Y_pred, print_stat, average, pos_label),
                confusion_matrix(TestY, test_Y_pred)]
    else:
        return [measure(TrainY, train_Y_pred, print_stat, average, pos_label),
                measure(DevY, dev_Y_pred, print_stat, average, pos_label),
                None,
                confusion_matrix(TestY, test_Y_pred)]
    '''
    return None
