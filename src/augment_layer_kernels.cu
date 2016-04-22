#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "augment_layer.h"
#include "cuda.h"
}

__global__ void augmentflip_kernel(int size, int w, int h, float *src, float *out)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= size) return;
    int x = id % w;
    id /= w;
    int y = id % h;

    float f = src[x+y*w];
    int out_index = (w-x-1)+y*w;
    out[out_index] = f;
}

__global__ void augmentflip_delta_kernel(int size, int w, int h, float ALPHA, float *src, float *out)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= size) return;
    int x = id % w;
    id /= w;
    int y = id % h;

    float f = src[x+y*w] * ALPHA;
    int out_index = (w-x-1)+y*w;
    out[out_index] += f;
}

extern "C" void augmentflip_gpu(int w, int h, float *src, float *dest)
{
    int size = w*h;
    augmentflip_kernel<<<cuda_gridsize(size), BLOCK>>>(size, w, h, src, dest);
    check_error(cudaPeekAtLastError());
}

extern "C" void augmentflip_delta_gpu(int w, int h, float ALPHA, float *src, float *dest)
{
    int size = w*h;
    augmentflip_delta_kernel<<<cuda_gridsize(size), BLOCK>>>(size, w, h, ALPHA, src, dest);
    check_error(cudaPeekAtLastError());
}


__device__ float get_pixel_kernel(float *image, int w, int h, int x, int y)
{
    if(x < 0 || x >= w || y < 0 || y >= h) return 0;
    return image[x + w*y];
}

__device__ float bilinear_interpolate_kernel(float *image, int w, int h, float x, float y)
{
    int ix = (int) floorf(x);
    int iy = (int) floorf(y);

    float dx = x - ix;
    float dy = y - iy;

    float val = (1-dy) * (1-dx) * get_pixel_kernel(image, w, h, ix, iy) +
        dy     * (1-dx) * get_pixel_kernel(image, w, h, ix, iy+1) +
        (1-dy) *   dx   * get_pixel_kernel(image, w, h, ix+1, iy) +
        dy     *   dx   * get_pixel_kernel(image, w, h, ix+1, iy+1);
    return val;
}


__global__ void augmentrotate_kernel(int size, int w, int h, float *src, float *out, float angle)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= size) return;
    int x = id % w;
    id /= w;
    int y = id % h;

    int cx = w/2;
    int cy = h/2;

    float rx = cos(angle)*(x-cx) - sin(angle)*(y-cy) + cx;
    float ry = sin(angle)*(x-cx) + cos(angle)*(y-cy) + cy;
    int out_index = x+y*w;
    out[out_index] = bilinear_interpolate_kernel(src, w, h, rx, ry);
}

__global__ void augmentrotate_delta_kernel(int size, int w, int h, float ALPHA, float *src, float *out, float angle)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= size) return;
    int x = id % w;
    id /= w;
    int y = id % h;

    int cx = w/2;
    int cy = h/2;

    float rx = cos(angle)*(x-cx) - sin(angle)*(y-cy) + cx;
    float ry = sin(angle)*(x-cx) + cos(angle)*(y-cy) + cy;
    int out_index = x+y*w;
    out[out_index] += ALPHA*bilinear_interpolate_kernel(src, w, h, rx, ry);

}

extern "C" void augmentrotate_gpu(int w, int h, float *src, float *dest, int ang)
{
    float radians = (float)(ang)*3.14159265/180.;
    int size = w*h;
    augmentrotate_kernel<<<cuda_gridsize(size), BLOCK>>>(size, w, h, src, dest, radians);
    check_error(cudaPeekAtLastError());
}

extern "C" void augmentrotate_delta_gpu(int w, int h, float ALPHA, float *src, float *dest, int ang)
{
    float radians = -(float)(ang)*3.14159265/180.;
    int size = w*h;
    augmentrotate_delta_kernel<<<cuda_gridsize(size), BLOCK>>>(size, w, h, ALPHA, src, dest, radians);
    check_error(cudaPeekAtLastError());
}





__global__ void forward_augment_flip_kernel(int n, int out_w, int out_h, int out_c,
                                               int gap, int w, int h,
                                               float *src, float *dest)
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

    int widx = x + out_w*(y + out_h*(c + b*out_c));
    float val = 0;
    if (y<h)
    {
        // copy
        int ridx = x + w*(y + h*(c + b*out_c));
        val = src[ridx];
    } else if (y>=h+gap)
    {
        // flip
        int ridx = (w-x-1) + w*((y-gap-h) + h*(c + b*out_c));
        val = src[ridx];
    }
    dest[widx] = val;
}

__global__ void backward_augment_flip_kernel(int n, int out_w, int out_h, int out_c,
                                             int gap, int w, int h,
                                             float *src, float *dest)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int x = id % w;
    id /= w;
    int y = id % h;
    id /= h;
    int c = id % out_c;
    id /= out_c;
    int b = id;

    int widx = x + y*w + c*w*h + b*out_c*w*h;
    int ridx = x + y*out_w + c*out_w*out_h + b*out_w*out_h*out_c;
    float dt = 0.5 * src[ridx];
    ridx = (w-x-1) + (y+gap+h)*out_w + c*out_w*out_h + b*out_w*out_h*out_c;
    dt += 0.5 * src[ridx];
    dest[widx] += dt;
}

extern "C" void augment_flip_gpu(int w, int h, int c, int batch, int gap,
                                 float *src, float *dest)
{
    int out_w = w;
    int out_h = 2*h + gap;
    int out_c = c;
    size_t n = out_h*out_w*out_c*batch;

    forward_augment_flip_kernel<<<cuda_gridsize(n), BLOCK>>>(n, out_w, out_h, out_c, gap, w, h,
                                                             src, dest);
    check_error(cudaPeekAtLastError());
}

extern "C" void augment_flip_delta_gpu(int w, int h, int c, int batch, int gap,
                                       float *src, float *dest)
{
    int out_w = w;
    int out_h = 2*h + gap;
    int out_c = c;
    size_t n = h*w*c*batch;

    backward_augment_flip_kernel<<<cuda_gridsize(n), BLOCK>>>(n, out_w, out_h, out_c, gap, w, h,
                                                             src, dest);
    check_error(cudaPeekAtLastError());
}



__global__ void augment_forward_kernel(int size, int w, int h, float *src, float *out, float angle, int flip, float scale)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= size) return;
    int x = id % w;
    id /= w;
    int y = id % h;

    int cx = w/2;
    int cy = h/2;

    float rx = scale*(cos(angle)*(x-cx) - sin(angle)*(y-cy)) + cx;
    float ry = scale*(sin(angle)*(x-cx) + cos(angle)*(y-cy)) + cy;
    rx = (flip ? (w-rx-1) : rx);
    int out_index = x+y*w;//(flip ? (w-x-1) : x) + y*w;

    int ix = (int)floorf(rx);
    int iy = (int)floorf(ry);
    float val = 0;
    if (!(ix < 0 || ix >= w || iy < 0 || iy >= h))
    {
        val = src[ix+iy*w];
    }
    out[out_index] = val;//bilinear_interpolate_kernel(src, w, h, rx, ry);
}


extern "C" void augment_forward_gpu(int w, int h, float *src, float *dest, float angle, int flip, float scale)
{
    float radians = (float)(angle)*3.14159265/180.;
    int size = w*h;
    augment_forward_kernel<<<cuda_gridsize(size), BLOCK>>>(size, w, h, src, dest, radians, flip, scale);
    check_error(cudaPeekAtLastError());
}

__global__ void augment_backward_kernel(int size, int w, int h, float *src, float *out, float angle, int flip, float scale, float ALPHA)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= size) return;
    int x = id % w;
    id /= w;
    int y = id % h;

    int cx = w/2;
    int cy = h/2;

    float rx = scale*(cos(angle)*(x-cx) - sin(angle)*(y-cy)) + cx;
    float ry = scale*(sin(angle)*(x-cx) + cos(angle)*(y-cy)) + cy;
    rx = (flip ? (w-rx-1) : rx);

    int ix = (int)floorf(rx);
    int iy = (int)floorf(ry);
    if ((ix < 0 || ix >= w || iy < 0 || iy >= h)) return;

    int out_index = ix+iy*w;
    out[out_index] += ALPHA * src[x+y*w];
}


extern "C" void augment_backward_gpu(int w, int h, float ALPHA, float *src, float *dest, float angle, int flip, float scale)
{
    float radians = (float)(angle)*3.14159265/180.;
    int size = w*h;
    augment_backward_kernel<<<cuda_gridsize(size), BLOCK>>>(size, w, h, src, dest, radians, flip, scale, ALPHA);
    check_error(cudaPeekAtLastError());
}


__constant__ float c_radians[32];
__constant__ float c_scales[32];
__constant__ int c_flips[32];


__global__ void augment_backward_max_kernel(int size, int w, int h, int out_c, int out_w, int out_h, int gap,
                                            float *src, float *dest, int* indexes)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= size) return;

    int x = id % out_w;
    id /= out_w;
    int y = id % out_h;
    id /= out_h;
    int c = id % out_c;

    int widx = x + y*out_w + c*out_w*out_h;
    int max_i = indexes[widx];
    float dt = src[widx];
    dest[max_i] += dt;
}

__global__ void augment_forward_max_kernel(int size, int w, int h, int c, int out_w, int out_h, int gap,
                                           float *src, float *dest, int* indexes, int n_aug)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= size) return;
    int x = id % out_w;
    id /= out_w;
    int y = id % out_h;
    id /= out_h;
    int ch = id % c;

    int cx = out_w/2;
    int cy = out_h/2;

    int widx = x + y*out_w + ch*out_w*out_h;
    int ridx = x + y*w + ch*w*h;
    int rbase = ch*w*h;
    float max = src[ridx];
    int max_i = ridx;
    int i;
    for (i=0;i<n_aug;i++)
    {
        rbase += gap*out_w + out_w*out_h;
        float rx = c_scales[i]*(cos(c_radians[i])*(x-cx) - sin(c_radians[i])*(y-cy)) + cx;
        float ry = c_scales[i]*(sin(c_radians[i])*(x-cx) + cos(c_radians[i])*(y-cy)) + cy;
        rx = (c_flips[i] ? (w-rx-1) : rx);

        int ix = (int)floorf(rx);
        int iy = (int)floorf(ry);
        if (ix < 0 || ix >= out_w || iy < 0 || iy >= out_h) continue;
        ridx = rbase+ix+iy*out_w;
        //float val = bilinear_interpolate_kernel(src+rbase, out_w, out_h, rx, ry);

        float val = src[ridx];
        max_i = (val > max) ? ridx : max_i;
        max   = (val > max) ? val  : max;
    }

    dest[widx] = max;
    indexes[widx] = max_i;
}


extern "C" void augment_forward_max_gpu(int w, int h, int c, int out_w, int out_h, int gap,
                                        float *src, float *dest, int* indexes,
                                        int n_aug,
                                        float* angles, int* flips, float* scales)
{
    float radians[32];
    float scales_[32];
    int i;
    for (i=0;i<n_aug;i++)
    {
        radians[i] = (float)(angles[i])*3.14159265/180.;
        if (flips[i]==0)
            radians[i] = -radians[i];
        scales_[i] = 1.f / scales[i];
    }

    cudaMemcpyToSymbol(c_radians, radians, n_aug*sizeof(float));
    cudaMemcpyToSymbol(c_scales, scales_, n_aug*sizeof(float));
    cudaMemcpyToSymbol(c_flips, flips, n_aug*sizeof(int));

    int size = out_w*out_h*c;
    augment_forward_max_kernel<<<cuda_gridsize(size), BLOCK>>>(size, w, h, c, out_w, out_h, gap, src, dest, indexes, n_aug);
    check_error(cudaPeekAtLastError());
}
extern "C" void augment_backward_max_gpu(int w, int h, int c, int out_w, int out_h, int gap,
                                        float *src, float *dest, int* indexes)
{
    int size = out_w*out_h*c;
    augment_backward_max_kernel<<<cuda_gridsize(size), BLOCK>>>(size, w, h, c, out_w, out_h, gap, src, dest, indexes);
    check_error(cudaPeekAtLastError());
}



__global__ void augment_backward_split_kernel(int size, int out_w, int out_h, int out_c,
                                            float *src, float *dest, int* indexes)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= size) return;

    int x = id % out_w;
    id /= out_w;
    int y = id % out_h;
    id /= out_h;
    int c = id % out_c;

    int widx = x + y*out_w + c*out_w*out_h;
    int ridx = indexes[widx];
    float dt = src[widx];
    dest[ridx] += dt;
}

__global__ void augment_forward_split_kernel(int size, int w, int h, int c, int out_w, int out_h, int gap,
                                           float *src, float *dest, int* indexes, int n_aug)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= size) return;
    int x = id % out_w;
    id /= out_w;
    int y = id % out_h;
    id /= out_h;
    int ch = id % (c*(1+n_aug)); // channel in output
    int chf = ch % c; // channel in input
    int augi = ch /c; // augmentation index

    int cx = out_w/2;
    int cy = out_h/2;

    int widx = x + y*out_w + ch*out_w*out_h;
    int rbase = chf*w*h;
    int ridx;

    if (augi==0)
    {
        // augmentation, just copy
        ridx = x + y*w + chf*w*h;
    } else
    {
        rbase += (gap*out_w + out_w*out_h) * augi;
        int i = augi-1;
        float rx = c_scales[i]*(cos(c_radians[i])*(x-cx) - sin(c_radians[i])*(y-cy)) + cx;
        float ry = c_scales[i]*(sin(c_radians[i])*(x-cx) + cos(c_radians[i])*(y-cy)) + cy;
        rx = (c_flips[i] ? (w-rx-1) : rx);

        int ix = (int)floorf(rx);
        int iy = (int)floorf(ry);
        if (ix < 0 || ix >= out_w || iy < 0 || iy >= out_h) return;
        ridx = rbase+ix+iy*out_w;
    }
    float v = src[ridx];
    dest[widx] = v;
    indexes[widx] = ridx;
}


extern "C" void augment_forward_split_gpu(int w, int h, int c, int out_w, int out_h, int gap,
                                        float *src, float *dest, int* indexes,
                                        int n_aug,
                                        float* angles, int* flips, float* scales)
{
    float radians[32];
    float scales_[32];
    int i;
    for (i=0;i<n_aug;i++)
    {
        radians[i] = (float)(angles[i])*3.14159265/180.;
        if (flips[i]==0)
            radians[i] = -radians[i];
        scales_[i] = 1.f / scales[i];
    }
    int out_c = c*(1+n_aug);

    cudaMemcpyToSymbol(c_radians, radians, n_aug*sizeof(float));
    cudaMemcpyToSymbol(c_scales, scales_, n_aug*sizeof(float));
    cudaMemcpyToSymbol(c_flips, flips, n_aug*sizeof(int));

    int size = out_w*out_h*out_c;
    augment_forward_split_kernel<<<cuda_gridsize(size), BLOCK>>>(size, w, h, c, out_w, out_h, gap, src, dest, indexes, n_aug);
    check_error(cudaPeekAtLastError());
}
extern "C" void augment_backward_split_gpu(int out_w, int out_h, int out_c,
                                        float *src, float *dest, int* indexes)
{
    int size = out_w*out_h*out_c;
    augment_backward_split_kernel<<<cuda_gridsize(size), BLOCK>>>(size, out_w, out_h, out_c, src, dest, indexes);
    check_error(cudaPeekAtLastError());
}

