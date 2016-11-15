"""
Long Short Term Memory for character level entity classification
"""
import sys
import argparse
import numpy as np
from xman import *
from utils import *
from autograd import *
from copy import deepcopy
import pickle
np.random.seed(0)

class LSTM(object):
    """
    Long Short Term Memory + Feedforward layer
    Accepts maximum length of sequence, input size, number of hidden units and output size
    """
    def __init__(self, max_len, in_size, num_hid, out_size):
        self.my_xman= self._build(max_len, in_size, num_hid, out_size) #DO NOT REMOVE THIS LINE. Store the output of xman.setup() in this variable

    def _build(self, max_len, in_size, num_hid, out_size, n=10):
        xm = XMan()
        #TODO: define your model here
        #define inputs x_1, x_2, ..., x_{max_len}
        for i in xrange(1, max_len+1):
            setattr(xm, 'x'+str(i), f.input(name='x'+str(i), default=np.random.rand(n, in_size)))
        xm.y = f.input(name='y', default=np.random.rand(n, out_size))

        # define parameters for gates and candidate weights
        w_i =  (6.0 / (in_size + num_hid)) ** 0.5
        u_i =  (6.0 / (num_hid + num_hid)) ** 0.5
        for char in ['i','f','o','c']: #input, forget, output, candidate
            setattr(xm, 'w'+char, f.param(name='w'+char, default=np.random.uniform(-w_i, w_i, (in_size, num_hid))))
            setattr(xm, 'u'+char, f.param(name='u'+char, default=np.random.uniform(-u_i, u_i, (num_hid, num_hid))))
            setattr(xm, 'b'+char, f.param(name='b'+char, default=np.random.uniform(-0.1,0.1,num_hid)))
        # Initialize c_0 and h_0, it is defined as inputs, because it depends on the batch size
        xm.c0 = f.input(name='c0', default=np.zeros((n, num_hid)))
        xm.h0 = f.input(name='h0', default=np.zeros((n, num_hid)))
        # calculating over time
        for t in xrange(1, max_len+1):
            # i_t = sigmoid(x_t * W_i + h_{t-1}*U_i + b_i)
            setattr(xm, 'i'+str(t), f.sigmoid(
                f.matrix_mul(getattr(xm,'x'+str(t)), xm.wi) +
                f.matrix_mul(getattr(xm,'h'+str(t-1)), xm.ui) +
                xm.bi))
            setattr(xm, 'f' + str(t), f.sigmoid(
                f.matrix_mul(getattr(xm, 'x' + str(t)), xm.wf) +
                f.matrix_mul(getattr(xm, 'h' + str(t - 1)), xm.uf) +
                xm.bf))
            setattr(xm, 'o' + str(t), f.sigmoid(
                f.matrix_mul(getattr(xm, 'x' + str(t)), xm.wo) +
                f.matrix_mul(getattr(xm, 'h' + str(t - 1)), xm.uo) +
                xm.bo))
            setattr(xm, 'c_tilt' + str(t), f.tanh(
                f.matrix_mul(getattr(xm, 'x' + str(t)), xm.wc) +
                f.matrix_mul(getattr(xm, 'h' + str(t - 1)), xm.uc) +
                xm.bc))

            setattr(xm, 'c'+str(t), f.ele_mul(getattr(xm, 'f'+str(t)), getattr(xm,'c'+str(t-1)))
                    + f.ele_mul(getattr(xm, 'i'+str(t)), getattr(xm, 'c_tilt'+str(t))) )
            setattr(xm, 'h'+str(t), f.ele_mul(getattr(xm, 'o'+str(t)), f.tanh(getattr(xm, 'c'+str(t)))))

        a = (6.0 / (num_hid + out_size)) ** 0.5
        xm.w2 = f.param('w2', default=np.random.uniform(-a,a,(num_hid, out_size)))
        xm.b2 = f.param('b2', default=np.random.uniform(-0.1,0.1,out_size))

        xm.o = f.relu(f.matrix_mul(getattr(xm, 'h'+str(t)), xm.w2) + xm.b2)
        xm.p = f.softMax(xm.o)
        xm.loss = f.crossEnt(xm.p, xm.y)
        return xm.setup()

def main(params, check_grad=False, test=False):
    epochs = params['epochs']
    max_len = params['max_len']
    num_hid = params['num_hid']
    batch_size = params['batch_size']
    dataset = params['dataset']
    init_lr = params['init_lr']
    output_file = params['output_file']

    # load data and preprocess
    dp = DataPreprocessor()
    data = dp.preprocess('%s.train'%dataset, '%s.valid'%dataset, '%s.test'%dataset)
    # minibatches
    mb_train = MinibatchLoader(data.training, batch_size, max_len, 
           len(data.chardict), len(data.labeldict))
    mb_valid = MinibatchLoader(data.validation, len(data.validation), max_len, 
           len(data.chardict), len(data.labeldict), shuffle=False)
    mb_test = MinibatchLoader(data.test, len(data.test), max_len, 
           len(data.chardict), len(data.labeldict), shuffle=False)
    # build
    print "building lstm..."
    lstm = LSTM(max_len,mb_train.num_chars,num_hid,mb_train.num_labels)
    #TODO CHECK GRADIENTS HERE
    if check_grad:
        epsilon = 1e-4
        print "Checking gradients..."
        my_xman = lstm.my_xman
        wengert_list = my_xman.operationSequence(my_xman.loss)
        print wengert_list
        value_dict = my_xman.inputDict()
        print 'input keys:', value_dict.keys()
        ad = Autograd(my_xman)
        value_dict = ad.eval(wengert_list, value_dict)
        print 'eval keys:', value_dict.keys()
        gradients = ad.bprop(wengert_list, value_dict, loss=np.float_(1.))
        print 'grad keys:', gradients.keys()
        # for key in ['b0','b1','w0','w1']:
        #     new_value_dict = value_dict.
    # train
    print "training..."
    # get default data and params
    my_xman = lstm.my_xman
    value_dict = my_xman.inputDict()
    lr = init_lr
    ad = Autograd(my_xman)
    wengert_list = my_xman.operationSequence(my_xman.loss)
    min_val_loss = float('Inf')
    opt_value_dict = {}
    print "Wengert List:", wengert_list
    if not test:
        for i in range(epochs):
            for (idxs,e,l) in mb_train:
                #prepare the input and do a fwd-bckwd pass over it and update the weights
                prepare_input(e, l, value_dict, max_len, num_hid)
                value_dict = ad.eval(wengert_list, value_dict)
                gradients = ad.bprop(wengert_list, value_dict, loss=np.float_(1.))
                #print "Training loss:", value_dict['loss']
                for key in gradients:
                    if my_xman.isParam(key):
                        value_dict[key] -= lr * gradients[key]

            # validate
            for (idxs,e,l) in mb_valid:
                #prepare the input and do a fwd pass over it to compute the loss
                prepare_input(e, l, value_dict, max_len, num_hid)
                value_dict = ad.eval(wengert_list, value_dict)
                print "%sth Epoch Validation loss" % i, value_dict['loss']
                # compare current validation loss to minimum validation loss
                if value_dict['loss'] < min_val_loss:
                    print "Update min val loss and store new params"
                    min_val_loss = value_dict['loss']
                    opt_value_dict = deepcopy(value_dict)
        print "Training done"
        print "Save optimized value dict into disk"
        pickle.dump(opt_value_dict, open('lstm_opt_dict', 'w'))
        print "done"
    if not test:
        value_dict = opt_value_dict
    if test:
        value_dict = pickle.load(open('mlp_opt_dict', 'r'))
    for (idxs,e,l) in mb_test:
        # prepare input and do a fwd pass over it to compute the output probs
        prepare_input(e, l, value_dict, max_len, num_hid)
        value_dict = ad.eval(wengert_list, value_dict)
        print "Test loss:", value_dict['loss']
        true = np.argmax(l, axis=1)
        predict = np.argmax(value_dict['p'], axis=1)
        precision = float(np.sum(true == predict)) / len(true)
        print "Precision:", precision

    # ensure that these are in the same order as the test input
    np.save(output_file, value_dict['p'])

def prepare_input(e, l, value_dict, max_len, num_hid):
    # e: N * M * V, need to be X_1 ~ X_m with N * V
    # l: N * C
    e, l = np.array(e), np.array(l)
    for i in xrange(1, max_len+1):
        value_dict['x'+str(i)] = e[:,i-1,:]
    value_dict['y'] = l
    n = e.shape[0]
    value_dict['c0'] = np.zeros((n, num_hid))
    value_dict['h0'] = np.zeros((n, num_hid))





if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_len', dest='max_len', type=int, default=10)
    parser.add_argument('--num_hid', dest='num_hid', type=int, default=50)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)
    parser.add_argument('--dataset', dest='dataset', type=str, default='tiny')
    parser.add_argument('--epochs', dest='epochs', type=int, default=15)
    parser.add_argument('--init_lr', dest='init_lr', type=float, default=0.5)
    parser.add_argument('--output_file', dest='output_file', type=str, default='output')
    params = vars(parser.parse_args())
    main(params)
