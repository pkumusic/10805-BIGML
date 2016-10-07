#!/usr/bin/env bash
javac LR.java
cat ../../data/dbpedia_16fall/abstract.tiny.train | java -Xmx128m LR 10000 0.5 0.1 1 44925 ../../data/dbpedia_16fall/abstract.tiny.train
#java -Xmx128m LR abc 10000 0.5 0.1 1 44925 test