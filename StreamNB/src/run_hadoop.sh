#!/usr/bin/env bash
javac -cp hadoop-core-1.0.1.jar:. *.java
jar cfe NB_train_hadoop.jar run *.class
hadoop jar NB_train_hadoop.jar ../RCV1/RCV1.very_small_train.txt out 5
