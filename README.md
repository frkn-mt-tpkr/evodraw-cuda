# EvoDraw: CUDA-Accelerated Evolutionary Image Reconstruction

EvoDraw is a high-performance GPU engine that reconstructs target images using geometric primitives (triangles, rectangles, circles) via Genetic Algorithms and Simulated Annealing.

## Technical Highlights
* **Hardware Optimization:** Written in C and CUDA, specifically optimized for resource-constrained GPUs (e.g., Nvidia MX330, sm_61 architecture).
* **Regional Undo Algorithm:** Eliminates PCIe bottlenecks by avoiding full-frame RAM-to-VRAM transfers. Calculates regional SAD (Sum of Absolute Differences) and updates only affected bounding boxes.
* **Simulated Annealing:** Dynamically scales primitive dimensions and alpha transparency over 500,000 iterations, transitioning from broad strokes to fine-grained details.
* **Safe Memory Management:** Implements clamped boundary limits to prevent illegal memory access and segmentation faults during massive parallel executions.

## Compilation
Use the following command to compile on Windows with MSVC and CUDA Toolkit:

```cmd
nvcc -arch=sm_61 -allow-unsupported-compiler -Xcompiler "/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH" main.cu -o evodraw_gpu