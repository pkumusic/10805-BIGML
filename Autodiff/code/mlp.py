"""
Multilayer Perceptron for character level entity classification
"""
import argparse
import numpy as np
from xman import *
from utils import *
from autograd import *

np.random.seed(0)

class MLP(object):
    """
    Multilayer Perceptron
    Accepts list of layer sizes [in_size, hid_size1, hid_size2, ..., out_size]
    """
    def __init__(self, layer_sizes):
        self.my_xman = self._build() # DO NOT REMOVE THIS LINE. Store the output of xman.setup() in this variable

    def _build(self):
        x = XMan()
        #TODO define your model here
        return x.setup()

def main(params):
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


    print "done"

    # train
    print "training..."
    # get default data and params
    value_dict = mlp.my_xman.inputDict()
    lr = init_lr
    for i in range(epochs):
        for (idxs,e,l) in mb_train:
            #TODO prepare the input and do a fwd-bckwd pass over it and update the weights

        # validate
        for (idxs,e,l) in mb_valid:
            #TODO prepare the input and do a fwd pass over it to compute the loss

        #TODO compare current validation loss to minimum validation loss
        # and store params if needed
    print "done"

    for (idxs,e,l) in mb_test:
        # prepare input and do a fwd pass over it to compute the output probs
        
    #TODO save probabilities on test set
    # ensure that these are in the same order as the test input
    #np.save(output_file, ouput_probabilities)

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
