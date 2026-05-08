## ToyRoCC Software Stack. 
This repository includes the following:
* The necessary minimal code to generate a Spike library for extending Spike to support the accelerator and test its functionality.
* A minimal C code to trigger the accelerator through custom instructions.

### Licensing
New BF16 matmul, attention, transformer, Tiny-BERT, and related benchmark
workloads added for this project are licensed under Apache-2.0. See
`../LICENSE.Apache-2.0` and `../NOTICE`.

Existing upstream ToyRoCC and third-party files keep their original licenses.

### Layout
* `src/`: C test and benchmark sources.
* `build/`: generated `.riscv` and `.dump` artifacts.
* `workloads/bare/`: FireMarshal bare-metal workload configs and host-init.
* `workloads/linux/`: FireMarshal Linux workload configs and host-init.
* `overlay/`: files injected into Linux rootfs images.

### Bare Matmul Workloads
* `workloads/bare/matmul-simple.json`: boots `systolic_matmul_simple_bfloat16_weight_stationary.riscv`.
* `workloads/bare/matmul-random.json`: boots `systolic_matmul_random_bfloat16_weight_stationary.riscv`.
* `workloads/bare/matmul-simple-ws.json`: boots `systolic_matmul_simple_bfloat16_weight_stationary.riscv`.
* `workloads/bare/matmul-random-ws.json`: boots `systolic_matmul_random_bfloat16_weight_stationary.riscv`.
* `workloads/bare/matmul-benchmark.json`: boots `systolic_matmul_benchmark_bfloat16_weight_stationary.riscv`.
* `workloads/bare/matmul-simple-2x2.json`: boots `systolic_matmul_simple_bfloat16_weight_stationary_2x2.riscv`.
* `workloads/bare/matmul-random-2x2.json`: boots `systolic_matmul_random_bfloat16_weight_stationary_2x2.riscv`.
* `workloads/bare/matmul-simple-2x2-ws.json`: boots `systolic_matmul_simple_bfloat16_weight_stationary_2x2.riscv`.
* `workloads/bare/matmul-random-2x2-ws.json`: boots `systolic_matmul_random_bfloat16_weight_stationary_2x2.riscv`.
* `workloads/bare/matmul-benchmark-2x2.json`: boots `systolic_matmul_benchmark_bfloat16_weight_stationary_2x2.riscv`.
* `workloads/linux/matmul-smoke-2x2.json`: boots Buildroot and runs the 2x2 GEMM smoke script.

### Bare Softmax And Attention Workloads
* `workloads/bare/softmax-smoke.json`: boots `softmax_norm_test.riscv`.
* `workloads/bare/softmax-rows-bf16.json`: boots `softmax_rows_bf16_test.riscv`.
* `workloads/bare/attention-bf16.json`: boots `attention_bf16_test.riscv`.

### Expected Verilator Output:
```sh
# This is a real RTL simulation.
azeer@azeer:~/git/chipyard/sims/verilator$ ./simulator-chipyard.harness-DummyToyRoCCConfig ../../generators/toyrocc/software/basic.riscv
[UART] UART0 is here (stdin/stdout).
Selam!
```

### Expected Spike Output:
```sh
# This is NOT a real RTL simulation.
# Just testing functionality.
azeer@azeer:~/git/chipyard/generators/toyrocc/software$ spike --extlib=libtoyrocc/libtoyrocc.so --extension=toy_rocc basic.riscv
Selam!
```
