#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "shrinkmax_layer.h"
#include "cuda.h"
}

__global__ void forward_shrinkmax_layer_kernel(int n, int w, int h, int out_w, int out_h, int out_c,
                                               int gap, int splits, int inputs,
                                               float *input, float *output, int *indexes)
{

    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int x = id % out_w;
    id /= out_w;
    int y = id % out_h;
    id /= out_h;
    int c = id % out_c;
    id /= out_c;
    int b = id;

    int ridx = x + y*w + c*w*h /*+ c*(splits-1)*out_w*gap*/ + b*inputs;
    int step = w*out_h+w*gap;
    int i;
    float max = -INFINITY;
    int max_i = -1;
    for (i=0;i<splits;++i)
    {
        float val = input[ridx];
        max_i = (val > max) ? ridx : max_i;
        max   = (val > max) ? val  : max;
        ridx += step;
    }
    int widx = x + y*out_w + c*out_w*out_h + b*out_c*out_w*out_h;
    output[widx] = max;
    indexes[widx] = max_i;
}

__global__ void backward_shrinkmax_layer_kernel(int n, int out_w, int out_h, int out_c,
                                                int gap, int splits, int inputs,
                                                float *delta, float *output, int *indexes)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int x = id % out_w;
    id /= out_w;
    int y = id % out_h;
    id /= out_h;
    int c = id % out_c;
    id /= out_c;
    int b = id;

    int widx = x + y*out_w + c*out_w*out_h + b*out_c*out_w*out_h;
    int max_i = indexes[widx];
    float dt = delta[widx];
    output[max_i] += dt;
}

extern "C" void forward_shrinkmax_layer_gpu(layer l, network_state state)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.out_c;

    size_t n = h*w*c*l.batch;

    forward_shrinkmax_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, l.w, l.h, l.out_w, l.out_h, l.out_c, l.gap, l.index, l.inputs, state.input, l.output_gpu, l.indexes_gpu);
    check_error(cudaPeekAtLastError());
}

extern "C" void backward_shrinkmax_layer_gpu(layer l, network_state state)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.out_c;

    size_t n = h*w*c*l.batch;

    backward_shrinkmax_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, l.out_w, l.out_h, l.out_c, l.gap, l.index, l.inputs, l.delta_gpu, state.delta, l.indexes_gpu);
    check_error(cudaPeekAtLastError());
}

