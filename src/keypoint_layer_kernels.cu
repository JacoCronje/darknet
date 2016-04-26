#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "keypoint_layer.h"
#include "cuda.h"
}

__global__ void keypoint_kernel(int size, int w, int h, float tx, float ty, float *dest)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= size) return;
    int out_index = id;
    float x = (float)(id % w) / w;
    id /= w;
    float y = (float)(id % h) / h;

    float f = (tx-x)*(tx-x) + (ty-y)*(ty-y);

    f = exp(-(f) / 0.03);

    dest[out_index] = f;
}

extern "C" void keypoint_gpu(int w, int h, float tx, float ty, float *dest)
{
    int size = w*h;
    keypoint_kernel<<<cuda_gridsize(size), BLOCK>>>(size, w, h, tx, ty, dest);
    check_error(cudaPeekAtLastError());
}
