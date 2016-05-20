#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "compact_layer.h"
#include "cuda.h"
}

__global__ void compact_backward_max_kernel(int size, float *src, float *dest, int* indexes)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= size) return;

    int ridx = indexes[id];
    float dt = src[id];
    dest[ridx] += dt;
}

__global__ void compact_forward_max_kernel(int size, int splits, float *src, float *dest, int* indexes)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= size) return;
    int ridx = id;
    int widx = id;

    float max = src[ridx];
    int max_i = ridx;
    int i;
    for (i=1;i<splits;i++)
    {
        ridx += size;
        float val = src[ridx];
        max_i = (val > max) ? ridx : max_i;
        max   = (val > max) ? val  : max;
    }
    dest[widx] = max;
    indexes[widx] = max_i;
}


extern "C" void compact_forward_max_gpu(int w, int h, int c, int splits, float *src, float *dest, int *indexes)
{
    int size = w*h*c/splits;
    compact_forward_max_kernel<<<cuda_gridsize(size), BLOCK>>>(size, splits, src, dest, indexes);
    check_error(cudaPeekAtLastError());
}

extern "C" void compact_backward_max_gpu(int w, int h, int c, int splits, float *src, float *dest, int *indexes)
{
    int size = w*h*c/splits;
    compact_backward_max_kernel<<<cuda_gridsize(size), BLOCK>>>(size, src, dest, indexes);
    check_error(cudaPeekAtLastError());
}

__global__ void compact_backward_padd_kernel1(int size, int iw, int ih, int ic, float *src, float *dest)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= size) return;

    float val = src[id];//*0.5f;
    dest[id] += val;
  //  dest[(id+iw*ih)%size] += val;
}
__global__ void compact_backward_padd_kernel2(int size, int iw, int ih, int ic, float *src, float *dest)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= size) return;

    float val = src[id];//*0.5f;
   // dest[id] += val;
    dest[(id+iw*ih)%size] += val;
}

__global__ void compact_forward_padd_kernel(int size, int iw, int ih, int ic, float *src, float *dest)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= size) return;

    float val = src[id];
    val += src[(id+iw*ih)%size];
    dest[id] = val;//*0.5f;
}


extern "C" void compact_forward_padd_gpu(int w, int h, int c, float *src, float *dest)
{
    int size = w*h*c;
    compact_forward_padd_kernel<<<cuda_gridsize(size), BLOCK>>>(size, w, h, c, src, dest);
    check_error(cudaPeekAtLastError());
}

extern "C" void compact_backward_padd_gpu(int w, int h, int c, float *src, float *dest)
{
    int size = w*h*c;
    compact_backward_padd_kernel1<<<cuda_gridsize(size), BLOCK>>>(size, w, h, c, src, dest);
    check_error(cudaPeekAtLastError());
    compact_backward_padd_kernel2<<<cuda_gridsize(size), BLOCK>>>(size, w, h, c, src, dest);
    check_error(cudaPeekAtLastError());
}


__global__ void compact_backward_pmax_kernel(int size, int iw, int ih, int ic, float *src, float *dest, int* indexes)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= size) return;

    int ridx = indexes[id];
    float dt = src[id];
    dest += ridx;
    atomicAdd(dest, dt);
    //dest[ridx] += dt;
}

__global__ void compact_forward_pmax_kernel(int size, int iw, int ih, int ic, float *src, float *dest, int* indexes)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= size) return;

    int ridx = id;
    int widx = id;

    float max = src[ridx];
    int max_i = ridx;
    ridx = (ridx+iw*ih)%size;
    float val = src[ridx];
    max_i = (val > max) ? ridx : max_i;
    max   = (val > max) ? val  : max;
    dest[widx] = max;
    indexes[widx] = max_i;
}

extern "C" void compact_forward_pmax_gpu(int w, int h, int c, float *src, float *dest, int *indexes)
{
    int size = w*h*c;
    compact_forward_pmax_kernel<<<cuda_gridsize(size), BLOCK>>>(size, w, h, c, src, dest, indexes);
    check_error(cudaPeekAtLastError());
}

extern "C" void compact_backward_pmax_gpu(int w, int h, int c, float *src, float *dest, int *indexes)
{
    int size = w*h*c;
    compact_backward_pmax_kernel<<<cuda_gridsize(size), BLOCK>>>(size, w, h, c, src, dest, indexes);
    check_error(cudaPeekAtLastError());
}





