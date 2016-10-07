#!/usr/bin/env bash
cat train | java -Xmx128m LR 10000 0.5 0.1 1 44925 test