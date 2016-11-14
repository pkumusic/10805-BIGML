"""
Multilayer Perceptron for character level entity classification
"""
import argparse
import numpy as np
from xman import *
from utils import *
from autograd import *
from copy import deepcopy
import pickle

np.random.seed(0)

class MLP(object):
    """
    Multilayer Perceptron
    Accepts list of layer sizes [in_size, hid_size1, hid_size2, ..., out_size]
    """
    def __init__(self, layer_sizes):
        self.my_xman = self._build(layer_sizes) # DO NOT REMOVE THIS LINE. Store the output of xman.setup() in this variable

    def _build(self, layer_sizes, n=10):
        xm = XMan()
        #TODO define your model here
        # 0 -> 1 -> 2
        #     w0,b0  w1,b1
        # o0 -> o1-> o2
        print "Number of layers including input and output: ", len(layer_sizes)
        xm.o0 = f.input(name='o0', default=np.random.rand(n, layer_sizes[0])) # N * in_size
        xm.y = f.input(name='y', default=np.random.rand(n, layer_sizes[-1]))
        for i in xrange(len(layer_sizes)-1):
            din, dout = layer_sizes[i], layer_sizes[i+1]
            a = (6.0 / (din + dout)) ** 0.5
            setattr(xm, 'w'+str(i), f.param(name='w'+str(i), default=np.random.uniform(-a, a, (din, dout))))
            setattr(xm, 'b'+str(i), f.param(name='b'+str(i), default=np.random.uniform(-0.1,0.1,dout)))
            setattr(xm, 'o'+str(i+1), f.relu(f.matrix_mul(getattr(xm, 'o'+str(i)), getattr(xm, 'w'+str(i)))+getattr(xm, 'b'+str(i))))
        xm.p = f.softMax(getattr(xm, 'o'+str(i+1)))
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
    print "building mlp..."
    mlp = MLP([max_len*mb_train.num_chars,num_hid,mb_train.num_labels])
    #TODO CHECK GRADIENTS HERE
    if check_grad:
        epsilon = 1e-4
        print "Checking gradients..."
        my_xman = mlp.my_xman
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

    print "done"

    # train
    print "training..."
    # get default data and params
    my_xman = mlp.my_xman
    value_dict = my_xman.inputDict()
    lr = init_lr
    #lr = 0.05
    ad = Autograd(my_xman)
    wengert_list = my_xman.operationSequence(my_xman.loss)
    print "Wengert List:", wengert_list

    min_val_loss = float('Inf')
    opt_value_dict = {}
    if not test:
        # First Validation for debug
        for (idxs, e, l) in mb_valid:
            # prepare the input and do a fwd pass over it to compute the loss
            e, l = np.array(e), np.array(l)
            e = e.reshape(e.shape[0], -1)
            value_dict['o0'], value_dict['y'] = e, l
            value_dict = ad.eval(wengert_list, value_dict)
            print "%sth Epoch Validation loss" % -1, value_dict['loss']


        for i in range(epochs):
            #if i % 10 == 0:
            #    print "Training epoch: ", i
            for (idxs,e,l) in mb_train:
                #prepare the input and do a fwd-bckwd pass over it and update the weights
                e,l = np.array(e), np.array(l)
                e = e.reshape(e.shape[0], -1)
                #print 'shape of mini-batch input:', e.shape
                #print 'shape of mini-batch labels:', l.shape
                value_dict['o0'], value_dict['y'] = e, l
                value_dict = ad.eval(wengert_list, value_dict)
                #print "Training loss:", value_dict['loss']
                gradients  = ad.bprop(wengert_list, value_dict, loss=np.float_(1.))
                for key in gradients:
                    if my_xman.isParam(key):
                        value_dict[key] -= lr * gradients[key]


            # validate
            for (idxs,e,l) in mb_valid:
                # prepare the input and do a fwd pass over it to compute the loss
                e, l = np.array(e), np.array(l)
                e = e.reshape(e.shape[0], -1)
                value_dict['o0'], value_dict['y'] = e, l
                value_dict = ad.eval(wengert_list, value_dict)
                print "%sth Epoch Validation loss" %i, value_dict['loss']
                # compare current validation loss to minimum validation loss
                if value_dict['loss'] < min_val_loss:
                    print "Update min val loss and store new params"
                    min_val_loss = value_dict['loss']
                    opt_value_dict = deepcopy(value_dict)
        print "Training done"
        print "Save optimized value dict into disk"
        pickle.dump(opt_value_dict, open('mlp_opt_dict', 'w'))

    # Read the optimized params
    if not test:
        value_dict = opt_value_dict
    if test:
        value_dict = pickle.load(open('mlp_opt_dict','r'))
    for (idxs,e,l) in mb_test:
        # prepare the input and do a fwd pass over it to compute the loss
        e, l = np.array(e), np.array(l)
        e = e.reshape(e.shape[0], -1)
        value_dict['o0'], value_dict['y'] = e, l
        value_dict = ad.eval(wengert_list, value_dict)
        print "Test loss:", value_dict['loss']
        true = np.argmax(l, axis=1)
        predict = np.argmax(value_dict['p'], axis=1)
        precision = float(np.sum(true==predict)) / len(true)
        print "Precision:", precision
        # prepare input and do a fwd pass over it to compute the output probs
        
    # save probabilities on test set
    # ensure that these are in the same order as the test input
    np.save(output_file, value_dict['p'])

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
