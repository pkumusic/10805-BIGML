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
from pyspark import SparkConf, SparkContext


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

    """ terminate """
    sc.stop()


if __name__ == '__main__':
    main(sys.argv)

