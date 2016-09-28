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

def get_hash(accum, (label, count)):
	accum[label] = count
	return accum

def calc_prob(accum, x, alpha=1):
	(id, word, label, count, label_docs, label_words, docs_count, num_words, num_labels) = x
	if accum == 0:
		# Add prior
		accum += math.log((label_docs[label]+alpha/num_labels)/ (docs_count+alpha))
	accum += math.log((count+alpha/num_words)/(label_words[label]+alpha))
	return accum

def calc_max_prob(accum, x):
	if accum == 0.0:
		accum = ("", float('-Inf'))
	(max_label, max) = accum
	((id, label), prob) = x
	if prob > max:
		accum = (label, prob)
	return accum

def calc_accuracy(accum, x):
	(id, label, labels, docs_count) = x
	if label in labels.split(','):
		accum += 1/docs_count
	return accum



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
	event1 =  Group(label_word_pair, by=lambda x:x, reducingTo=ReduceToCount(), combiningTo=ReduceToCount()) \
			| Map(by=lambda ((label,word),count): (label, word, count))

	## Add 0 to label word pairs
	labels = Map(event1, by=lambda (label,word,count): (label)) | Distinct()
	words  = Map(event1, by=lambda (label,word,count): (word))  | Distinct()
	event0 = Join(Jin(labels, by=lambda x:1), Jin(words, by=lambda x:1))	\
				   | Map(by=lambda (label,word):(label, word, 0))
	event = Union(event1, event0) | Group(by=lambda (label,word,count):(label,word), reducingTo=ReduceTo(int,by=lambda accum,(label,word,count):accum+count)) | Map(by=lambda ((label,word),count):(label,word,count))

	## Global
	# train (label, count)
	label_words_count = Group(event, by=lambda (label,word,count):label, reducingTo=ReduceTo(int,by=lambda accum,(label,word,count):accum+count))
	# ('label_words', hashmap)
	label_words_hash = Group(label_words_count, by=lambda (label, count):'label_words', reducingTo=ReduceTo(dict, by=lambda accum, (label,count): get_hash(accum, (label, count))))
	# train (label, count)
	label_docs_count = Flatten(train_lines, by=label_doc) | Group(by=lambda x:x, reducingTo=ReduceToCount(), combiningTo=ReduceToCount())
	# ('label_docs', hashmap)
	label_docs_hash = Group(label_docs_count, by=lambda (label, count):'label_docs', reducingTo=ReduceTo(dict, by=lambda accum, (label,count): get_hash(accum, (label, count))))
	# docs_count
	docs_count = Group(label_docs_count, by=lambda x:"docs_count", reducingTo=ReduceTo(int,by=lambda accum, (label,count):accum+count))
	num_words = Group(words, by=lambda x:"words", reducingTo=ReduceToCount(), combiningTo=ReduceToCount())
	num_labels = Group(labels, by=lambda x: "labels", reducingTo=ReduceToCount(), combiningTo=ReduceToCount())

	# test (id, labels, doc)
	test_fields = Map(test_lines, by=lambda line: line.strip().split('\t'))

	# test (id, word)
	test_id_word = Flatten(test_fields, by=id_word)

	# (id, word, label, count)
	joined = Join(Jin(test_id_word, by=lambda (id,word):word), Jin(event, by=lambda (label, word, count):word)) \
			| Map(by=lambda ((id, word1),(label,word2,count)):(id,word1,label,count))

	# (id, word, label, count, label_docs, label_words, docs_count, num_words, num_labels)
	infos = Augment(joined, sideviews=[label_docs_hash, label_words_hash, docs_count, num_words, num_labels], loadedBy=lambda x,y,z,a,b:(GPig.onlyRowOf(x),GPig.onlyRowOf(y),GPig.onlyRowOf(z),GPig.onlyRowOf(a),GPig.onlyRowOf(b))) \
						| Map(by=lambda ((id, word, label, count), ((n, label_docs), (n1,label_words), (n2,docs_count), (n3, num_words), (n4, num_labels))): (id, word, label, count, label_docs, label_words, docs_count, num_words, num_labels))
	#infos1 = Augment(joined, sideview=label_docs_hash, loadedBy=lambda x:GPig.onlyRowOf(x)) | Map(by=lambda ((id,word,label,count),(_,label_docs)): (id,word,label,count,label_docs))
	#infos2 = Augment(infos1, sideview=label_words_hash, loadedBy=lambda x: GPig.onlyRowOf(x)) | Map(by=lambda ((id, word, label, count, label_docs), (_, label_words)): (id, word, label, count, label_docs, label_words))
	#infos = Augment(infos2, sideview=docs_count, loadedBy=lambda x:GPig.onlyRowOf(x)) | Map(by=lambda ((id, word, label, count, label_docs, label_words), (_, docs_count)): (id, word, label, count, label_docs, label_words, docs_count))

	# Calculating
	# (id, label, prob)
	calc1 = Group(infos, by=lambda (id, word, label, count, label_docs, label_words, docs_count, num_words, num_labels): (id, label), reducingTo=ReduceTo(float, by=lambda accum, x:calc_prob(accum, x)))
	# (id, label, max_prob)
	output = Group(calc1, by=lambda ((id, label), prob): id, reducingTo=ReduceTo(float, by=lambda accum, x:calc_max_prob(accum, x))) | Map(by=lambda (id, (label, prob)): (id,label,prob))

	# (id, labels)
	gold = Map(test_fields, by=lambda (id,labels,doc): (id, labels))

	# (id, label, labels, docs_count)
	compare = Join(Jin(output, by=lambda (id, label, max_prob):id), Jin(gold, by=lambda (id,labels):id)) | Map(by=lambda (((id1, label, max_prob)),(id2,labels)): (id1, label, labels))

	accuracy = Augment(compare, sideview=docs_count, loadedBy=lambda x:GPig.onlyRowOf(x)) | Map(by=lambda ((id,label,labels),(_, docs_count)): (id, label, labels, docs_count)) \
					| Group(by=lambda x:'Accuracy', reducingTo=ReduceTo(float, by=lambda accum, x: calc_accuracy(accum,x)))

# always end like this
if __name__ == "__main__":
    NB().main(sys.argv)

# supporting routines can go here
