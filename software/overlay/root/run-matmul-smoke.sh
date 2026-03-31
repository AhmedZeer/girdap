#!/bin/bash

echo "TEST STARTED !"

echo "*****************TEST RESULTS*************" #> $OUT_FILE_NAME

echo "========Matmul Simple=========\n"
/root/systolic_matmul_simple.riscv #>> $OUT_FILE_NAME

echo "\n\n\n========Matmul Random=========\n"
/root/systolic_matmul_random.riscv #>> $OUT_FILE_NAME

echo "\n\n\n========Matmul Benchmark=========\n"
/root/systolic_matmul_benchmark.riscv #>> $OUT_FILE_NAME

echo "*****************END SCRIPT*************" #> $OUT_FILE_NAME

poweroff -f
