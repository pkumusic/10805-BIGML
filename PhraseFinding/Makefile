run:
	time spark-submit --driver-memory 2G pf.py 1980 data/unigram_war.txt data/bigram_war.txt data/stopwords.txt 100 50 5 100 > result.txt 2> result.log

war1960:
	spark-submit --driver-memory 2G pf.py 1960 data/unigram_war.txt data/bigram_war.txt data/stopwords.txt 100 50 5 50 > war1960.txt
	python gen_wc.py war1960.txt war1960.png

apple1980:
	spark-submit --driver-memory 2G pf.py 1980 data/unigram_apple.txt data/bigram_apple.txt data/stopwords.txt 100 50 5 50 > apple1980.txt
	python gen_wc.py apple1980.txt apple1980.png

handin:
	tar cvf handin.tar result.txt result.log pf.py
