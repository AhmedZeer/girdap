![image](docs/static/banner.png)
## TL;DR
**Atik** is an open-source AI Accelerator Hardware.

*What makes it so special?* Here 👇

✅ **Attention** mechanism smelted directly into silicon 
✅ Prototyped end-to-end on **FPGA** (AWS F2) 
✅ Benchmarked against **PyTorch**-based workloads 
✅ Built on the **RocketChip** architecture (RISC-V) 
✅ Native **BF16** support 
✅ Up to **100×** speedup on vanilla attention 
✅ Up to **80×** speedup on TinyBERT 
✅ Up to **30×** speedup on GPT-2 prefill 

- Don't believe benchmark results? Click to watch the playlist
- Want to simulate it locally? Check out this video.
- Have access to AWS F2 instances and want to reproduce the PyTorch benchmarks? Follow this guide.
- Want to build your own from scratch? Follow this playlist. *(Coming soon)*

*From here upon nerdy people can continue reading :)*


## Why Atik ?
There's definitely a growing interest in academia around Attention accelerators, rivaling the interest in systolic hardware. But honestly, a lot of the current research feels a bit too theoretical. Much of it relies purely on C++ simulators like gem5, lacking time-accurate simulation or a proper VLSI flow. Otherwise, it tends to be closed-source, or a standalone ASIC implementation that doesn't easily integrate with standard CPU workloads. 

The mature, open-source DNN accelerators we already have—like NVDLA and Gemmini—are starting to feel a bit legacy. They were beautifully built for the needs of their time, optimizing for CNNs and vision tasks, but they don't support Attention mechanisms. Plus, they focus on quantized datatypes like INT8, which just isn't ideal for modern day-to-day applications that rely so heavily on BF16. 

Standard vector units simply aren't cutting it anymore. With transformers being adopted everywhere, we really need systolic arrays working right alongside Softmax modules to finally tackle those heavy attention workloads. 

To bring it all together: what the open-source hardware community truly needs right now is a dedicated Attention and MatMul accelerator. It needs to support BF16 natively, sit on top of a robust computer architecture like RocketChip, be easily benchmarked against modern PyTorch workloads, and be fully ready for FPGA prototyping.

This is the gap **Atik** is trying to fill. A modern opensource Tightly-Coupled AI accelerator. FPGA-prototypable. VLSI-verified. Preparing for someone to tape it out! 

#### Mature architecture
#### Answering today's needs
#### Open source LLM hardware

## Architecture

## Time & Cycle Accurate Simulation On FPGA

## Benchmark Results

## Benchmark Videos

## VLSI Flow

## Tutorial Videos

## Project Timeline

## Acknowledgment