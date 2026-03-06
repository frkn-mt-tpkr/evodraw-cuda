#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _MSVC_ENABLE_CUDA_NON_BLOCKING_SYNCHRONIZATION

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define CHECK(call) { const cudaError_t error = call; if (error != cudaSuccess) { printf("HATA: %s:%d, Mesaj: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); exit(1); } }

__device__ float sign(float x1, float y1, float x2, float y2, float x3, float y3) {
    return (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3);
}

__device__ bool is_in_triangle(float px, float py, float x1, float y1, float x2, float y2, float x3, float y3) {
    float d1 = sign(px, py, x1, y1, x2, y2);
    float d2 = sign(px, py, x2, y2, x3, y3);
    float d3 = sign(px, py, x3, y3, x1, y1);
    bool has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    bool has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);
    return !(has_neg && has_pos);
}

__global__ void calculate_regional_sad_kernel(unsigned char* ref, unsigned char* canvas, int w, int h, int b_x, int b_y, int b_w, int b_h, unsigned long long* region_sad) {
    int x = b_x + blockIdx.x * blockDim.x + threadIdx.x;
    int y = b_y + blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 0 || x >= w || y < 0 || y >= h) return;

    if (x >= b_x && x < b_x + b_w && y >= b_y && y < b_y + b_h) {
        int i = (y * w + x) * 3;
        int diff = abs((int)ref[i] - (int)canvas[i]) +
                   abs((int)ref[i+1] - (int)canvas[i+1]) +
                   abs((int)ref[i+2] - (int)canvas[i+2]);
        atomicAdd(region_sad, (unsigned long long)diff);
    }
}

__global__ void draw_shape_kernel(unsigned char* canvas, int w, int h, int type, float p1, float p2, float p3, float p4, float p5, float p6,
                                  unsigned char r, unsigned char g, unsigned char b, float alpha, int b_x, int b_y, int b_w, int b_h) {
    int x = b_x + blockIdx.x * blockDim.x + threadIdx.x;
    int y = b_y + blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 0 || x >= w || y < 0 || y >= h) return;

    if (x >= b_x && x < b_x + b_w && y >= b_y && y < b_y + b_h) {
        bool inside = false;
        if (type == 0) {
            float d2 = (x-p1)*(x-p1) + (y-p2)*(y-p2);
            if (d2 <= p3*p3) inside = true;
        } else if (type == 1) {
            if (x >= p1 && x < p1+p3 && y >= p2 && y < p2+p4) inside = true;
        } else if (type == 2) {
            if (is_in_triangle((float)x, (float)y, p1, p2, p3, p4, p5, p6)) inside = true;
        }

        if (inside) {
            int i = (y * w + x) * 3;
            canvas[i]   = (unsigned char)(r * alpha + canvas[i] * (1.0f - alpha));
            canvas[i+1] = (unsigned char)(g * alpha + canvas[i+1] * (1.0f - alpha));
            canvas[i+2] = (unsigned char)(b * alpha + canvas[i+2] * (1.0f - alpha));
        }
    }
}

__global__ void restore_region_kernel(unsigned char* canvas, unsigned char* best, int w, int h, int b_x, int b_y, int b_w, int b_h) {
    int x = b_x + blockIdx.x * blockDim.x + threadIdx.x;
    int y = b_y + blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 0 || x >= w || y < 0 || y >= h) return;

    if (x >= b_x && x < b_x + b_w && y >= b_y && y < b_y + b_h) {
        int i = (y * w + x) * 3;
        canvas[i] = best[i];
        canvas[i+1] = best[i+1];
        canvas[i+2] = best[i+2];
    }
}

#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN(a,b) ((a) < (b) ? (a) : (b))

int main() {
    srand((unsigned int)time(NULL));
    int w, h, c;
    unsigned char *h_img = stbi_load("referans.png", &w, &h, &c, 3);
    if (!h_img) return 1;

    int img_size = w * h * 3;
    unsigned char *d_ref, *d_canvas, *d_best;
    unsigned long long *d_reg_sad, h_reg_sad;

    CHECK(cudaMalloc((void**)&d_ref, img_size));
    CHECK(cudaMalloc((void**)&d_canvas, img_size));
    CHECK(cudaMalloc((void**)&d_best, img_size));
    CHECK(cudaMalloc((void**)&d_reg_sad, sizeof(unsigned long long)));

    CHECK(cudaMemcpy(d_ref, h_img, img_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_best, 0, img_size));
    CHECK(cudaMemcpy(d_canvas, d_best, img_size, cudaMemcpyDeviceToDevice));

    int MAX_ITER = 500000;
    printf("EvoDraw\n");

    for (int iter = 0; iter < MAX_ITER; iter++) {
        float progress = (float)iter / MAX_ITER;
        float alpha = 0.5f * (1.0f - progress) + 0.05f;
        int max_dim = (int)(120 * (1.0f - progress)) + 5;

        int type = rand() % 3;
        float p1, p2, p3, p4, p5, p6;
        int b_x, b_y, b_w, b_h;

        if (type == 0) {
            int cx = rand() % w;
            int cy = rand() % h;
            int r = (rand() % max_dim) / 2 + 2;
            p1 = cx; p2 = cy; p3 = r;
            int min_x = cx - r, min_y = cy - r, max_x = cx + r, max_y = cy + r;
            b_x = MAX(0, min_x);
            b_y = MAX(0, min_y);
            b_w = MIN(w - 1, max_x) - b_x + 1;
            b_h = MIN(h - 1, max_y) - b_y + 1;
        } else if (type == 1) {
            int rx = rand() % w;
            int ry = rand() % h;
            int rw = (rand() % max_dim) + 2;
            int rh = (rand() % max_dim) + 2;
            p1 = rx; p2 = ry; p3 = rw; p4 = rh;
            b_x = MAX(0, rx);
            b_y = MAX(0, ry);
            b_w = MIN(w - 1, rx + rw - 1) - b_x + 1;
            b_h = MIN(h - 1, ry + rh - 1) - b_y + 1;
        } else {
            int tx1 = rand() % w;
            int ty1 = rand() % h;
            int tx2 = tx1 + (rand() % (max_dim * 2) - max_dim);
            int ty2 = ty1 + (rand() % (max_dim * 2) - max_dim);
            int tx3 = tx1 + (rand() % (max_dim * 2) - max_dim);
            int ty3 = ty1 + (rand() % (max_dim * 2) - max_dim);
            p1 = tx1; p2 = ty1; p3 = tx2; p4 = ty2; p5 = tx3; p6 = ty3;
            int min_x = MIN(tx1, MIN(tx2, tx3));
            int min_y = MIN(ty1, MIN(ty2, ty3));
            int max_x = MAX(tx1, MAX(tx2, tx3));
            int max_y = MAX(ty1, MAX(ty2, ty3));
            b_x = MAX(0, min_x);
            b_y = MAX(0, min_y);
            b_w = MIN(w - 1, max_x) - b_x + 1;
            b_h = MIN(h - 1, max_y) - b_y + 1;
        }

        if (b_w <= 0 || b_h <= 0) continue;

        dim3 block(16, 16);
        dim3 grid((b_w + 15) / 16, (b_h + 15) / 16);

        h_reg_sad = 0;
        cudaMemcpy(d_reg_sad, &h_reg_sad, sizeof(unsigned long long), cudaMemcpyHostToDevice);
        calculate_regional_sad_kernel<<<grid, block>>>(d_ref, d_best, w, h, b_x, b_y, b_w, b_h, d_reg_sad);
        cudaDeviceSynchronize();
        cudaMemcpy(&h_reg_sad, d_reg_sad, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        unsigned long long old_err = h_reg_sad;

        unsigned char r_col = rand()%256, g_col = rand()%256, b_col = rand()%256;
        draw_shape_kernel<<<grid, block>>>(d_canvas, w, h, type, p1, p2, p3, p4, p5, p6, r_col, g_col, b_col, alpha, b_x, b_y, b_w, b_h);
        cudaDeviceSynchronize();

        h_reg_sad = 0;
        cudaMemcpy(d_reg_sad, &h_reg_sad, sizeof(unsigned long long), cudaMemcpyHostToDevice);
        calculate_regional_sad_kernel<<<grid, block>>>(d_ref, d_canvas, w, h, b_x, b_y, b_w, b_h, d_reg_sad);
        cudaDeviceSynchronize();
        cudaMemcpy(&h_reg_sad, d_reg_sad, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        unsigned long long new_err = h_reg_sad;

        if (new_err < old_err) {
            CHECK(cudaMemcpy(d_best, d_canvas, img_size, cudaMemcpyDeviceToDevice));
            if (iter % 1000 == 0) printf("\n[%d] Iyilesme! Tip: %d, Alpha: %.2f", iter, type, alpha);
        } else {
            restore_region_kernel<<<grid, block>>>(d_canvas, d_best, w, h, b_x, b_y, b_w, b_h);
            cudaDeviceSynchronize();
        }
        if (iter % 500 == 0) { printf("."); fflush(stdout); }
    }

    unsigned char *h_final = (unsigned char*)malloc(img_size);
    CHECK(cudaMemcpy(h_final, d_best, img_size, cudaMemcpyDeviceToHost));
    stbi_write_png("evodraw_final.png", w, h, 3, h_final, w * 3);
    printf("\nBitti!\n");
    return 0;
}