#!/usr/bin/env python
# encoding: utf-8

"""
@brief Phrase finding with spark
@param fg_year The year taken as foreground
@param f_unigrams The file containing unigrams
@param f_bigrams The file containing bigrams
@param f_stopwords The file containing stop words
@param w_info Weight of informativeness
@param w_phrase Weight of phraseness
@param n_workers Number of workers
@param n_outputs Number of top bigrams in the output
"""

import sys
from operator import add
from pyspark import SparkConf, SparkContext

def fg_word_count(grams, fg_year):
    grams = grams.filter(lambda (words, decade, count): int(decade) == fg_year).map(lambda (words, decade, count): (words, count))
    grams = grams.reduceByKey(add)
    return grams

def bg_word_count(grams, fg_year):
    grams = grams.filter(lambda (words, decade, count): int(decade) != fg_year).map(lambda (words, decade, count): (words, count))
    grams = grams.reduceByKey(add)
    return grams

def total_unique_word_count(grams):
    return grams.map(lambda (words, decade, count): (words, count)).groupByKey().count()

def main(argv):
    # parse args
    fg_year = int(argv[1])
    f_unigrams = argv[2]
    f_bigrams = argv[3]
    f_stopwords = argv[4]
    w_info = float(argv[5])
    w_phrase = float(argv[6])
    n_workers = int(argv[7])
    n_outputs = int(argv[8])

    """ configure pyspark """
    conf = SparkConf().setMaster('local[{}]'.format(n_workers))  \
                      .setAppName(argv[0])
    sc = SparkContext(conf=conf)
    
    # TODO: start your code here
    stopwords = sc.textFile(f_stopwords).cache()
    stopwords = set(stopwords.collect())
    # (unigram, decade, count)
    unigrams = sc.textFile(f_unigrams).map(lambda x:tuple(x.split('\t')))
    unigrams = unigrams.filter(lambda (unigram, decade, count): unigram not in stopwords)
    # ((word1, word2), decade, count)
    bigrams  = sc.textFile(f_bigrams).map(lambda x: x.split('\t'))
    bigrams  = bigrams.map(lambda (words, decade, count): ((words.split(' ')[0], words.split(' ')[1]), decade, count))
    bigrams  = bigrams.filter(lambda ((word1, word2), decade, count): word1 not in stopwords and word2 not in stopwords)
    # foreground word count 
    fgWordCount = fg_word_count(unigrams, fg_year)
    bgWordCount = bg_word_count(unigrams, fg_year)
    fgPhraseCount = fg_word_count(bigrams, fg_year)
    bgPhraseCount = bg_word_count(bigrams, fg_year)
    # Total count
    B = total_unique_word_count(bigrams)
    U = total_unique_word_count(unigrams)
    print "Number of unique bigrams", B
    print "Number of unique unigrams", U

    #res = fgPhraseCount.collect()
    #print res
    #print len(res)

    
    
    

    """ terminate """
    sc.stop()


if __name__ == '__main__':
    main(sys.argv)

