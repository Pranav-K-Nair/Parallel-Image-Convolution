# Parallelization of Image Convolution with Gaussian Filter

## Implementations

### 1. Full GUI App
An interactive app that let's you choose which algorithm to run for the filtering

**Instructions:**
```bash
make
./main_run
```

### 2. Sequential Gaussian Filter
A baseline implementation using traditional serial processing.

**Instructions:**
```bash
cd test
g++ sequential.cpp -o sequential
./sequential
```

### 3. OpenMP Implementation
Parallel implementation using OpenMP for multi-core CPU processing.

**Instructions:**
```bash
cd test
g++ -fopenmp openmp_filtering.cpp -o openmp_filtering
./openmp_filtering
```

### 4. Advanced CPU Implementation
Optimized CPU implementation with advanced compiler optimizations and SIMD instructions.

**Instructions:**
```bash
cd test
g++ -march=native -fopenmp -mavx2 -mfma advanced_cpu.cpp -o advanced_cpu
./advanced_cpu
```

### 5. CUDA - Global Memory Implementation
GPU-accelerated implementation using CUDA with global memory access patterns.

**Instructions:**
```bash
cd test
nvcc gaussian_global.cu -o gaussian_global
./cuda_global
```

## 6. CUDA - Shared Memory Implementation
GPU-accelerated implementation using CUDA with shared memory access patterns.

**Compilation:**
```bash
cd test
nvcc gaussian_shared.cu -o gaussian_shared
./cuda_global
```
