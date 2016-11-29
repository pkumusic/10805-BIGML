spark-submit --driver-memory 2G pf.py $1 data/unigram_war.txt data/bigram_war.txt data/stopwords.txt 100 1 5 50 > war$1.txt
python gen_wc.py war$1.txt war$1_info.png
