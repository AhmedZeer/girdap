## ToyRoCC Software Stack. 
This repository includes the following:
* The necessary minimal code to generate a Spike library for extending Spike to support the accelerator and test its functionality.
* A minimal C code to trigger the accelerator through custom instructions.

### Layout
* `src/`: C test and benchmark sources.
* `build/`: generated `.riscv` and `.dump` artifacts.
* `workloads/bare/`: FireMarshal bare-metal workload configs and host-init.
* `workloads/linux/`: FireMarshal Linux workload configs and host-init.
* `overlay/`: files injected into Linux rootfs images.

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
