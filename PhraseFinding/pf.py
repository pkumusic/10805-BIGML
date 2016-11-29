#!/usr/bin/env python
# encoding: utf-8
from __future__ import division
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
import math
from operator import add
from pyspark import SparkConf, SparkContext

def fg_word_count(grams, fg_year):
    grams = grams.filter(lambda (words, decade, count): decade == fg_year).map(lambda (words, decade, count): (words, count))
    grams = grams.reduceByKey(add)
    return grams

def bg_word_count(grams, fg_year):
    grams = grams.filter(lambda (words, decade, count): decade != fg_year).map(lambda (words, decade, count): (words, count))
    grams = grams.reduceByKey(add)
    return grams

def total_unique_word_count(grams):
    return grams.map(lambda (words, decade, count): (words, count)).groupByKey().count()

def total_word_count(wordCount):
    wordCount = wordCount.map(lambda (words, count): count).sum()
    return wordCount

def clear_none(x):
    (words, (c1, c2)) = x
    if c1 == None:
        c1 = 0
    if c2 == None:
        c2 = 0
    return (words, (c1,c2))

def add_stats(s):
    ((x,y), (stats, fxC, fyC)) = s
    if fxC == None:
        fxC = 0
    if fyC == None:
        fyC = 0
    stats['fxC'] = fxC
    stats['fyC'] = fyC
    return ((x,y), stats)

def calc_prob(x, B, U, fg_bigram_total, bg_bigram_total, fg_unigram_total, w_phrase, w_info):
    ((x,y), stats) = x
    Pfg = (stats['fC'] + 1) / (B + fg_bigram_total)
    Pbg = (stats['bC'] + 1) / (B + bg_bigram_total)
    Pfgx = (stats['fxC'] + 1) / (U + fg_unigram_total)
    Pfgy = (stats['fyC'] + 1) / (U + fg_unigram_total)
    phra = Pfg * math.log(Pfg/(Pfgx*Pfgy))
    info = Pfg * math.log(Pfg/Pbg)
    score = w_phrase * phra + w_info * info
    
    return ((x,y), score)

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
    unigrams = unigrams.map(lambda (word, decade, count): (word, int(decade), int(count)))
    unigrams = unigrams.filter(lambda (unigram, decade, count): unigram not in stopwords)
    # ((word1, word2), decade, count)
    bigrams  = sc.textFile(f_bigrams).map(lambda x: x.split('\t'))
    bigrams  = bigrams.map(lambda (words, decade, count): ((words.split(' ')[0], words.split(' ')[1]), decade, count))
    bigrams  = bigrams.map(lambda (word, decade, count): (word, int(decade), int(count)))
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
    fg_bigram_total = total_word_count(fgPhraseCount)
    bg_bigram_total = total_word_count(bgPhraseCount)
    fg_unigram_total = total_word_count(fgWordCount)
    print "Number of total foreground bigrams", fg_bigram_total
    print "Number of total backgroud bigrams", bg_bigram_total
    print "Number of total foreground unigrams", fg_unigram_total
    phraseCount = fgPhraseCount.fullOuterJoin(bgPhraseCount).map(lambda x: clear_none(x)).map(lambda (words, (fC, bC)): (words, {'fC':fC, 'bC':bC}))
    phraseCount = phraseCount.map(lambda ((x, y), stats): (x, (y, stats))).leftOuterJoin(fgWordCount).map(lambda (x, ((y,stats),fxC)): (y, (x, stats, fxC)))
    phraseCount = phraseCount.leftOuterJoin(bgWordCount).map(lambda (y, ((x,stats, fxC), fyC)): ((x,y), (stats, fxC, fyC))).map(lambda x: add_stats(x))
    phraseCount = phraseCount.map(lambda x: calc_prob(x, B, U, fg_bigram_total, bg_bigram_total, fg_unigram_total, w_phrase, w_info))
    ans = sorted(phraseCount.collect(), key=lambda ((x,y),score): score, reverse=True)
    for i in xrange(n_outputs):
        entry = ans[i]
        print entry[0][0] + '-' + entry[0][1] + ':' + str(entry[1])
    #res = fgPhraseCount.collect()
    #print res
    #print len(res)

    
    
    

    """ terminate """
    sc.stop()


if __name__ == '__main__':
    main(sys.argv)

