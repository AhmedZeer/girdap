#!/usr/bin/env bash

echo "TEST STARTED !"

OUT_FILE_NAME="softmax_test_output.txt"
echo "*****************TEST RESULTS*************" > $OUT_FILE_NAME

echo "========Softmax Smoke========="
/root/softmax_norm_test.riscv >> $OUT_FILE_NAME

cat $OUT_FILE_NAME
poweroff -f
