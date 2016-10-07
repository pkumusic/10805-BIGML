#!/usr/bin/env bash
javac LR.java
for((i=1;i<=20;i++));
do shuf trainData.txt;
done | java -Xmx128m LR 10000 0.5 0.1 20 1000 testData.txt