# CSCI596-Final-Project

## Team members:

- Jiayi Liu
- Hanyu Zheng

## GPU-Accelerated MapReduce Performance Analysis

### Project Overview

This project aims to analyze and compare the performance of MapReduce tasks executed on GPUs versus CPUs. Our primary goal is to benchmark the execution time differences between these two processing units, highlighting the efficiency and speed improvements offered by GPU processing.

### Key Objectives

- **Benchmarking MapReduce on GPUs**: Running MapReduce tasks on GPUs to evaluate their performance.
- **Comparison with CPU Performance**: Comparing these results with equivalent tasks performed on CPUs.

### Technologies Used

- **MapReduce Framework**: The core technology for processing large-scale data.
- **GPU Acceleration**: Leveraging GPU power to enhance processing speed.
- **Programming Languages & Libraries**: Cuda.

## Project Details: Dice Roll Simulation Using MapReduce on GPUs vs. CPUs

In our project, we conducted a simulation to compare the performance of a dice roll simulation executed on GPUs and CPUs. This practical application of our research provides an insightful perspective on the effectiveness of GPU acceleration.

### Experimental Design

- **Task**: Simulating the roll of a die 10 million times.
- **Objective**: To evaluate and compare the execution speed and efficiency on GPUs and CPUs.

### Results and Analysis

#### GPU Performance

- Probabilities:
  - Face 1: 16.632990%
  - Face 2: 16.622030%
  - Face 3: 16.701279%
  - Face 4: 16.705080%
  - Face 5: 16.665730%
  - Face 6: 16.672890%
- Execution Time: 4.810 seconds

#### CPU Performance

- Probabilities:
  - Face 1: 16.6651%
  - Face 2: 16.6632%
  - Face 3: 16.6686%
  - Face 4: 16.6711%
  - Face 5: 16.6629%
  - Face 6: 16.669%
- Execution Time: 6.62625 seconds

The GPU's superior performance can be attributed to its parallel processing capabilities, which allow for simultaneous computation of multiple operations. This contrasts with the CPU's sequential processing method, making the GPU more efficient for repetitive tasks like our dice roll simulation.

### Conclusions and Future Work

This experiment has demonstrated the clear advantages of using GPUs for certain types of computational tasks. Our future work will focus on further exploring the capabilities and potential of GPU-accelerated computing, including:

- Testing with a variety of GPU models to evaluate performance differences.
- Conducting in-depth analysis of scalability and efficiency across different GPU architectures.
- Optimizing MapReduce algorithms for better performance on GPU systems.
