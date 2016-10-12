#!/usr/bin/env bash
javac LR.java
for((i=1;i<=20;i++));
do gshuf ../../data/dbpedia_16fall/abstract.tiny.train;
done | java -Xmx128m LR 10000 0.5 0.1 20 10000 ../../data/dbpedia_16fall/abstract.tiny.test