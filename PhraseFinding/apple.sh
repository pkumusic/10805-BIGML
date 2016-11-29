spark-submit --driver-memory 2G pf.py $1 data/unigram_apple.txt data/bigram_apple.txt data/stopwords.txt 100 50 5 50 > apple$1.txt
python gen_wc.py apple$1.txt apple$1.png
