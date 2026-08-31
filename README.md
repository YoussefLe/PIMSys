# PIMSys: Bare-Metal Firmware and VLIW Microcode for SpMV

This repository contains the low-level software stack (written in Rust `#![no_std]`) executing on the host ARM processor, alongside the hardware kernels (microcode) injected directly into HBM2 memory for Processing-In-Memory (PIM) acceleration. 

This submodule operates in tandem with the main gem5+DRAMSys co-simulation environment to demonstrate the elimination of the Memory Wall for Sparse Matrix-Vector Multiplication (SpMV).

## 1. Hardware Design: PIM Microcode

The PIM execution units residing inside the memory banks rely on a highly optimized, custom instruction set. The mathematical kernels are defined in `src/kernel/spmv.rs`.

* **1-Cycle MAC Instruction:** To overcome Read-After-Write (RAW) data hazards typically found in standard multi-cycle accumulation pipelines, the standard fetch-decode-execute loop is replaced by an atomic `MAC` (Multiply-Accumulate) hardware instruction. This ensures deterministic, high-throughput calculation directly at the memory cell.
* **Strict VLIW Alignment:** The kernel is engineered using a strict Very Long Instruction Word (VLIW) paradigm. It consists of exactly 5 active instructions padded with 27 `NOP` instructions. This forces a rigid 32-byte physical footprint, which perfectly matches a single HBM2 cache line. This alignment guarantees that the kernel is loaded atomically via a single memory burst, maximizing TSV (Through-Silicon Via) bandwidth efficiency.

## 2. Software Design: Host Firmware Orchestration

The ARM CPU acts entirely as a control-plane orchestrator. During the PIM offloading phase, the CPU performs absolutely zero mathematical matrix computations.

* **NNZ-Based Load Balancing (`partition_csr`):** Standard geometric row-splitting leads to severe load imbalance ("hot banks") due to the irregular dispersion inherent to sparse matrices. To prevent this, the firmware dynamically divides the Compressed Sparse Row (CSR) matrix into 32 equal-weight computational blocks based strictly on the Number of Non-Zeros (NNZ). This algorithm guarantees symmetric traffic and utilization across all 32 independent HBM2 banks.
* **Deep Sleep Orchestration (`wfi`):** After dispatching the memory-mapped offloading triggers to the HBM2 controller, the ARM core issues a Wait-For-Interrupt (`wfi`) assembly instruction. This completely suspends the processor's clock, drastically reducing the SoC's energy footprint while the memory autonomously processes the sparse matrix.

## 3. Repository Structure

* `src/bin/vadd.rs`: The primary executable orchestrating the PIM-accelerated SpMV calculation.
* `src/bin/baseline.rs`: The control executable (No PIM) where the ARM CPU performs a standard sequential SpMV. This is utilized strictly to benchmark and demonstrate the CPU memory bottleneck.
* `src/kernel/`: Contains the microcode payloads and hardware instructions injected into the HBM2 memory arrays.

## 4. Binary Cross-Compilation

The build environment relies on the Rust `cargo` package manager to generate operating-system-less (bare-metal) executables for the ARMv8 (`aarch64`) architecture. 

Compile the PIM Accelerated version (32-bank HPC parallelism):

    cargo build --release --bin vadd --target aarch64-unknown-none

Compile the CPU Baseline version (Sequential comparison):

    cargo build --release --bin baseline --target aarch64-unknown-none
