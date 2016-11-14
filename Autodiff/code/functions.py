# some useful functions
import numpy as np
from xman import *


# some useful functions
# declare all operations here first

class f(XManFunctions):
    @staticmethod
    def square(a):
        return XManFunctions.registerDefinedByOperator('square',a)
    @staticmethod
    def relu(a):
        return XManFunctions.registerDefinedByOperator('relu',a)
    @staticmethod
    def crossEnt(x1, x2):
        return XManFunctions.registerDefinedByOperator('crossEnt',x1,x2)
    @staticmethod
    def softMax(a):
        return XManFunctions.registerDefinedByOperator('softMax',a)
    @staticmethod
    def matrix_mul(x1, x2):
        return XManFunctions.registerDefinedByOperator('matrix_mul',x1,x2)

# the functions that autograd.eval will use to evaluate each function,
# to be called with the functions actual inputs as arguments

def _crossEnt(p, t): # x1: True label
    assert p.shape == t.shape
    return -np.sum(t * np.log(p)) / t.shape[0]

def _softMax(x):
    x = x - np.max(x, axis=1).reshape(-1,1)
    x = np.exp(x)
    x = x / np.sum(x, axis=1).reshape(-1,1)
    #print x
    return x


EVAL_FUNS = {
    'add':      lambda x1,x2: x1+x2,
    'subtract': lambda x1,x2: x1-x2,
    'square':   np.square,
    'crossEnt': lambda p,t: _crossEnt(p, t),
    'softMax':  lambda p: _softMax(p),
    'relu': lambda a: a * (a>0),
    'matrix_mul': np.dot
    }

# the functions that autograd.bprop will use in reverse mode
# differentiation.  BP_FUNS[f] is a list of functions df1,....,dfk
# where dfi is used in propagating errors to the i-th input xi of f.
# Specifically, dfi is called with the ordinary inputs to f, with two
# additions: the incoming error, and the output of the function, which
# was computed by autograd.eval in the eval stage.  dfi will return
# delta * df/dxi [f(x1,...,xk)]
# 
# NOTE: Autograd has an optimization where if it finds a softMax op
# followed by crossEnt op, it combines the backward pass for both. So
# you only need to implement the BP_FUNS for the combined operation 
# crossEnt-softMax below.

def _derivAdd(delta,x1):
    if delta.shape!=x1.shape:
        # broadcast, sum along axis=0
        if delta.shape[1]!=x1.shape[0]:
            raise ValueError("Dimension Mismatch")
        return delta.sum(axis=0) #we sum the gradients over the batch
    else: return delta

BP_FUNS = {
    'add':              [lambda delta,out,x1,x2: _derivAdd(delta,x1),    lambda delta,out,x1,x2: _derivAdd(delta,x2)],
    'subtract':         [lambda delta,out,x1,x2: _derivAdd(delta,x1),    lambda delta,out,x1,x2: -_derivAdd(delta,x2)],
    'square':           [lambda delta,out,x : delta * 2.0 * x],
    'crossEnt-softMax': [lambda delta,out,p,y:  delta * (p-y) / p.shape[0],
                         lambda delta,out,p,y: 1], # Would not be used.
    'relu': [lambda delta,out,x: delta * np.ones(x.shape) * (x>0)],
    'matrix_mul': [lambda delta,out,x1,x2:np.dot(delta, x2.T),
                   lambda delta,out,x1,x2:np.dot(x1.T, delta)]
    }
