from __future__ import division
from guineapig import *
import math
__Author__ = 'Music'

# supporting routines can go here
def tokenize(doc):
    for tok in doc.strip().split():
        yield tok.lower().replace("\\W","")

def label_word(line):
    [labels, doc] = line.strip().split('\t')[1:]
    labels = labels.split(',')
    for token in tokenize(doc):
        for label in labels:
            yield (label,token)

def id_word((id, labels, doc)):
    for token in tokenize(doc):
        yield (id, token)

def label_doc(line):
    labels = line.strip().split('\t')[1].split(',')
    for label in labels:
        yield label

def get_hash(accum, label):
    accum[label] = accum.get(label, 0) + 1
    return accum

def get_hash1(accum, x):
    accum[x[0]] = accum.get(x[0], 0) + 1
    return accum

def calc_max_prob(accum, x):
    if accum == 0.0:
        accum = ("", float('-Inf'))
    (max_label, max) = accum
    (id, label, prob) = x
    if prob > max:
        accum = (label, prob)
    return accum

def calc_accuracy(accum, x):
    (id, label, labels, docs_count) = x
    if label in labels.split(','):
        accum += 1/docs_count
    return accum

def hash_label_count(accum, x):
    (label, word, count) = x
    accum[label] = count
    return accum

def get_priors(hashmap):
    sum = 0
    for key, value in hashmap.iteritems():
        sum += value
    for key, value in hashmap.iteritems():
        hashmap[key] = math.log(value/sum)
    return hashmap

def get_probs(x):
    (id, word, counter, label_words, num_words) = x
    label_probs = {}
    for key, value in label_words.iteritems():
        label_probs[key] = math.log((counter.get(key, 0) + 1) / (value + num_words))
    return (id, word, label_probs)

def get_probs2(x):
    (id, word, counter, label_words, num_words, label_priors) = x
    label_probs = {}
    for key, value in label_words.iteritems():
        label_probs[key] = math.log((counter.get(key, 0) + 1) / (value + num_words))
    return (id, label_probs, label_priors)


def yield_label_prob(x):
    (id, word, probs) = x
    for key, value in probs.iteritems():
        yield (id, key, value)

def get_total_prob(accum, x):
    (id, label_probs, label_priors) = x
    for label, prob in label_probs.iteritems():
        accum[label] = accum.get(label, label_priors[label]) + prob
    return accum

def get_max_prob(x):
    (id, dic) = x
    max_label = max(dic, key=lambda x:dic[x])
    max_prob = dic[max_label]
    return (id, max_label, max_prob)



#always subclass Planner
class NB(Planner):
    # params is a dictionary of params given on the command line.
    # e.g. trainFile = params['trainFile']
    params = GPig.getArgvParams()
    train_lines = ReadLines(params['trainFile'])
    test_lines  = ReadLines(params['testFile'])
    # (label, word)
    label_word_pair = Flatten(train_lines, by=label_word)
    # train (label, word, count)
    event =  Group(label_word_pair, by=lambda x:x, reducingTo=ReduceToCount(), combiningTo=ReduceToCount()) \
            | Map(by=lambda ((label,word),count): (label, word, count))
    # (word, {label:count})
    event_counter = Group(event, by=lambda (label, word, count):word, reducingTo=ReduceTo(dict, by=lambda accum, x:hash_label_count(accum, x)))
    ## Global
    num_words = Map(label_word_pair, by=lambda (label, word): word) | Distinct() | Group(by=lambda x: "num_words",
                                                                                         reducingTo=ReduceToCount(),
                                                                                   combiningTo=ReduceToCount())
    # ('label_words', hashmap)
    label_words_hash = Group(label_word_pair, by=lambda x:"label_words", reducingTo=ReduceTo(dict, by=lambda accum,x:get_hash1(accum, x)))
    # ('label_docs', hashmap)
    label_docs_hash = Flatten(train_lines, by=label_doc) | Group(by=lambda x:"label_docs", reducingTo=ReduceTo(dict, by=lambda accum,x:get_hash(accum, x)))
    label_priors = Map(label_docs_hash, by=lambda (_, hashmap): ("label_priors", get_priors(hashmap)))

    # test (id, word)
    test_fields = Map(test_lines, by=lambda line: line.strip().split('\t'))
    test_id_word = Flatten(test_fields, by=id_word)

    # (id, word, {label:count})
    joined = Join(Jin(test_id_word, by=lambda (id,word):word), Jin(event_counter, by=lambda (word, counter):word)) \
            | Map(by=lambda ((id, word1),(word2,counter)):(id,word1,counter))

    # (id, word, {label: count}, label_words, num_words, label_priors)
    infos = Augment(joined, sideviews=[label_words_hash, num_words, label_priors], loadedBy=lambda x,y,z:(GPig.onlyRowOf(x),GPig.onlyRowOf(y),GPig.onlyRowOf(z))) \
                        | Map(by=lambda ((id, word, counter), ((n1, label_words), (n2,num_words), (n3, label_priors))): (id, word, counter, label_words, num_words, label_priors))
    # id, label, prob
    output = Map(infos, by=lambda x:get_probs2(x)) | \
             Group(by=lambda (id, label_probs, label_priors): id, reducingTo=ReduceTo(dict, by=lambda accum, x:get_total_prob(accum, x))) |\
             Map(by=lambda x:get_max_prob(x))


    # output = Map(infos, by=lambda x:get_probs(x)) | Flatten(by=lambda x:yield_label_prob(x)) \
    #         | Group(by=lambda (id, label, prob):(id, label), reducingTo=ReduceTo(float, by=lambda accum,x:accum+x[2])) \
    #         | Augment(sideview=label_priors, loadedBy=lambda x:GPig.onlyRowOf(x)) \
    #         | Map(by=lambda (((id, label), prob), (_, priors)): (id, label, prob+priors[label])) \
    #         | Group(by=lambda (id, label, prob): id, reducingTo=ReduceTo(float, by=lambda accum, x:calc_max_prob(accum, x))) \
    #         | Map(by=lambda (id, (label, prob)): (id,label,prob))

    # Test Accuracy
    # gold = Map(test_fields, by=lambda (id,labels,doc): (id, labels))
    # compare = Join(Jin(output, by=lambda (id, label, max_prob):id), Jin(gold, by=lambda (id,labels):id)) | Map(by=lambda (((id1, label, max_prob)),(id2,labels)): (id1, label, labels))
    # test_docs_num = Group(test_lines, by=lambda x:'test_docs_count', reducingTo=ReduceToCount())
    # accuracy = Augment(compare, sideview=test_docs_num, loadedBy=lambda x:GPig.onlyRowOf(x)) | Map(by=lambda ((id,label,labels),(_, docs_count)): (id, label, labels, docs_count)) \
    # 				| Group(by=lambda x:'Accuracy', reducingTo=ReduceTo(float, by=lambda accum, x: calc_accuracy(accum,x)))

# always end like this
if __name__ == "__main__":
    NB().main(sys.argv) #

# supporting routines can go here
