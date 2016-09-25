#!/usr/bin/env bash
javac -cp hadoop-core-1.0.1.jar:. *.java
jar cfe WC.jar WordCount *.class
hadoop jar WC.jar ../RCV1/RCV1.very_small_train.txt out 1
