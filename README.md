# EvoDraw: CUDA-Accelerated Evolutionary Image Reconstruction

EvoDraw is a high-performance GPU engine that reconstructs target images using geometric primitives (triangles, rectangles, circles) via **Genetic Algorithms** and **Simulated Annealing**.

> **🔥 Built and optimized for legacy NVIDIA GPUs** — This project was developed on an **NVIDIA MX330** (Pascal, sm_61), a GPU whose driver and CUDA support has been officially discontinued by NVIDIA. Despite running on deprecated hardware with only 2 GB VRAM and 256 CUDA cores, EvoDraw achieves efficient real-time evolutionary rendering by minimizing memory transfers and maximizing parallel throughput through custom regional algorithms.

<p align="center">
  <img src="images/referans.png" width="380" alt="Reference Image"/>
  &nbsp;&nbsp;➜&nbsp;&nbsp;
  <img src="images/evodraw_final.png" width="380" alt="Reconstructed Image"/>
</p>
<p align="center"><em>Left: Reference input — Right: Reconstructed output (500,000 iterations)</em></p>

## ✨ Features

- **Legacy GPU Optimized** — Specifically engineered to run on discontinued NVIDIA GPUs (MX330, GT 1030, etc.) where official CUDA support has ended. Proves that deprecated hardware can still deliver impressive compute results with the right optimizations.
- **Full GPU Acceleration** — Written in C and CUDA for massively parallel execution, with careful attention to the limited resources of older architectures.
- **Regional Undo Algorithm** — Eliminates PCIe bottlenecks by avoiding full-frame RAM-to-VRAM transfers. Calculates regional SAD (Sum of Absolute Differences) and updates only affected bounding boxes — critical for low-bandwidth legacy GPUs.
- **Simulated Annealing** — Dynamically scales primitive dimensions and alpha transparency over 500,000 iterations, transitioning from broad strokes to fine-grained details.
- **Safe Memory Management** — Implements clamped boundary limits to prevent illegal memory access during massive parallel executions.
- **Multi-Primitive Support** — Circles, rectangles, and triangles with alpha blending.

## 📁 Project Structure

```
evodraw-cuda/
├── src/
│   └── main.cu            # Core CUDA source code
├── include/
│   ├── stb_image.h        # Image loading (stb)
│   └── stb_image_write.h  # Image writing (stb)
├── images/
│   ├── referans.png        # Reference input image
│   └── evodraw_final.png   # Generated output
├── .gitignore
└── README.md
```

## 🔧 Requirements

- **NVIDIA GPU** — Works on legacy/deprecated GPUs (tested on MX330, sm_61). Also compatible with modern GPUs.
- **CUDA Toolkit** (11.0 or later) — Compilation uses `-allow-unsupported-compiler` to bypass version restrictions on older hardware.
- **MSVC** (Visual Studio C++ compiler) on Windows
- **Git** (for cloning the repository)

## 🚀 Getting Started

### Clone the Repository

```bash
git clone https://github.com/frkn-mt-tpkr/evodraw-cuda.git
cd evodraw-cuda
```

### Compile

```cmd
nvcc -arch=sm_61 -allow-unsupported-compiler -Xcompiler "/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH" -I include src/main.cu -o evodraw_gpu
```

> **Note:** Replace `sm_61` with your GPU's compute capability. Common values: `sm_75` (Turing), `sm_86` (Ampere), `sm_89` (Ada Lovelace).

### Run

Place your reference image as `images/referans.png` and run:

```cmd
evodraw_gpu.exe
```

The output will be saved to `images/evodraw_final.png`.

## 🧬 How It Works

1. **Initialization** — The target image is loaded and transferred to GPU memory. A blank canvas is created.
2. **Shape Generation** — A random primitive (circle, rectangle, or triangle) is generated with random position, size, color, and alpha.
3. **Fitness Evaluation** — Regional SAD is calculated on the GPU using atomic operations, comparing only the affected bounding box region.
4. **Selection** — If the new shape reduces error, it's kept. Otherwise, the region is restored from the best canvas (Regional Undo).
5. **Annealing** — Over 500K iterations, shape sizes shrink and alpha decreases, moving from coarse to fine detail.

## 📊 Performance & Benchmarks

The engine has been rigorously tested under constrained environments. Below are the performance results measured on deprecated hardware:

### Test Environment
- **GPU:** NVIDIA GeForce MX330 (Mobile)
- **VRAM:** 2 GB GDDR5
- **Architecture:** Pascal (sm_61) - *Officially unsupported in newest CUDA releases*
- **CUDA Cores:** 256
- **CUDA Toolkit:** Built with 11.x (via `-allow-unsupported-compiler`)

### Results
- **Target Iterations:** 500,000
- **Execution Time:** ~2-3 minutes (highly dependent on image resolution and bounding box dimensions)
- **VRAM Utilization:** < 50 MB (Extremely efficient memory footprint)
- **GPU Utilization:** 95-100% (Consistent high parallel efficiency)

By avoiding full-frame SAD computations and instead processing only the affected bounding boxes for each candidate shape, EvoDraw achieves a massive iteration rate per second, turning a hardware limitation into an algorithmic optimization showcase.

## 🏚️ Why Legacy GPUs?

NVIDIA regularly discontinues driver and CUDA support for older GPU architectures. Cards like the **MX330**, **GT 1030**, and other Pascal-era mobile/desktop GPUs are no longer receiving updates and are often considered obsolete for GPGPU workloads.

EvoDraw was built as a challenge to that assumption. By carefully optimizing memory access patterns, avoiding unnecessary host-device transfers, and using region-based computation instead of full-frame processing, this project demonstrates that **deprecated hardware can still be leveraged for meaningful GPU-accelerated tasks**.

Key optimizations for legacy hardware:
- **Regional bounding-box computation** instead of full-frame SAD — reduces VRAM bandwidth pressure
- **Device-to-device memory copies** (`cudaMemcpyDeviceToDevice`) to avoid slow PCIe round-trips
- **Minimal memory footprint** — only 3 GPU buffers (reference, canvas, best) fit comfortably in 2 GB VRAM
- **Compiler compatibility workarounds** (`-allow-unsupported-compiler`) to build with modern toolchains on legacy targets