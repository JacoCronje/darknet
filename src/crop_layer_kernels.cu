#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "crop_layer.h"
#include "utils.h"
#include "cuda.h"
#include "image.h"
}

__device__ float get_pixel_kernel(float *image, int w, int h, int x, int y, int c)
{
    if(x < 0 || x >= w || y < 0 || y >= h) return 0;
    return image[x + w*(y + c*h)];
}

__device__ float3 rgb_to_hsv_kernel(float3 rgb)
{
    float r = rgb.x;
    float g = rgb.y; 
    float b = rgb.z;

    float h, s, v;
    float max = (r > g) ? ( (r > b) ? r : b) : ( (g > b) ? g : b);
    float min = (r < g) ? ( (r < b) ? r : b) : ( (g < b) ? g : b);
    float delta = max - min;
    v = max;
    if(max == 0){
        s = 0;
        h = -1;
    }else{
        s = delta/max;
        if(r == max){
            h = (g - b) / delta;
        } else if (g == max) {
            h = 2 + (b - r) / delta;
        } else {
            h = 4 + (r - g) / delta;
        }
        if (h < 0) h += 6;
    }
    return make_float3(h, s, v);
}

__device__ float3 hsv_to_rgb_kernel(float3 hsv)
{
    float h = hsv.x;
    float s = hsv.y; 
    float v = hsv.z;

    float r, g, b;
    float f, p, q, t;

    if (s == 0) {
        r = g = b = v;
    } else {
        int index = (int) floorf(h);
        f = h - index;
        p = v*(1-s);
        q = v*(1-s*f);
        t = v*(1-s*(1-f));
        if(index == 0){
            r = v; g = t; b = p;
        } else if(index == 1){
            r = q; g = v; b = p;
        } else if(index == 2){
            r = p; g = v; b = t;
        } else if(index == 3){
            r = p; g = q; b = v;
        } else if(index == 4){
            r = t; g = p; b = v;
        } else {
            r = v; g = p; b = q;
        }
    }
    r = (r < 0) ? 0 : ((r > 1) ? 1 : r);
    g = (g < 0) ? 0 : ((g > 1) ? 1 : g);
    b = (b < 0) ? 0 : ((b > 1) ? 1 : b);
    return make_float3(r, g, b);
}

__device__ float bilinear_interpolate_kernel(float *image, int w, int h, float x, float y, int c)
{
    int ix = (int) floorf(x);
    int iy = (int) floorf(y);

    float dx = x - ix;
    float dy = y - iy;

    float val = (1-dy) * (1-dx) * get_pixel_kernel(image, w, h, ix, iy, c) + 
        dy     * (1-dx) * get_pixel_kernel(image, w, h, ix, iy+1, c) + 
        (1-dy) *   dx   * get_pixel_kernel(image, w, h, ix+1, iy, c) +
        dy     *   dx   * get_pixel_kernel(image, w, h, ix+1, iy+1, c);
    return val;
}

__global__ void levels_image_kernel(float *image, float *rand, int batch, int c, int w, int h, int train, float saturation, float exposure, float translate, float scale, float shift)
{
    int size = batch * w * h;
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= size) return;
    int x = id % w;
    id /= w;
    int y = id % h;
    id /= h;
    float rshift = rand[0];
    float gshift = rand[1];
    float bshift = rand[2];
    float r0 = rand[16*id + 0];
    float r1 = rand[16*id + 1];
    float r2 = rand[16*id + 2];
    float r3 = rand[16*id + 3];

    saturation = r0*(saturation - 1) + 1;
    saturation = (r1 > .5) ? 1./saturation : saturation;
    exposure = r2*(exposure - 1) + 1;
    exposure = (r3 > .5) ? 1./exposure : exposure;

    size_t offset = id * h * w * c;
    image += offset;

    if (c==3)
    {
        float r = image[x + w*(y + h*0)];
        float g = image[x + w*(y + h*1)];
        float b = image[x + w*(y + h*2)];
        float3 rgb = make_float3(r,g,b);
        if(train){
            float3 hsv = rgb_to_hsv_kernel(rgb);
            hsv.y *= saturation;
            hsv.z *= exposure;
            rgb = hsv_to_rgb_kernel(hsv);
        } else {
            shift = 0;
        }
        image[x + w*(y + h*0)] = rgb.x*scale + translate + (rshift - .5)*shift;
        image[x + w*(y + h*1)] = rgb.y*scale + translate + (gshift - .5)*shift;
        image[x + w*(y + h*2)] = rgb.z*scale + translate + (bshift - .5)*shift;
    } else
    {
        for (int cc=0;cc<c;cc++)
        {
            float r = image[x + w*(y + h*cc)];
            float g = r;
            float b = r;
            float3 rgb = make_float3(r,g,b);
            if(train){
                float3 hsv = rgb_to_hsv_kernel(rgb);
                hsv.y *= saturation;
                hsv.z *= exposure;
                rgb = hsv_to_rgb_kernel(hsv);
            } else {
                shift = 0;
            }
            image[x + w*(y + h*cc)] = rgb.x*scale + translate + (rshift - .5)*shift;
        }
    }

}

__constant__ float c_kernel[16*8];

__global__ void forward_crop_blurX_kernel(float *input, float *rand, int size, int c, int h, int w, float *output)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= size) return;

    int count = id;
    int x = id % w;
    id /= w;
    id /= h;
    id /= c;
    int b = id;

    float r9 = rand[16*b + 9];

    int i;
    int px = -7;
    int kidx = 16*(int)(r9*7);
    float out = 0;
    float num = 0;
    for (i=0;i<15;i++)
    {
        if (px+x>=0 && px+x<w)
        {
            out += output[count+px] * c_kernel[kidx];
            num += c_kernel[kidx];
        }
        px++;
        kidx++;
    }

    output[count] = out/num;
}

__global__ void forward_crop_blurY_kernel(float *input, float *rand, int size, int c, int h, int w, float *output)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= size) return;

    int count = id;
    id /= w;
    int y = id % h;
    id /= h;
    id /= c;
    int b = id;

    float r9 = rand[16*b + 9];

    int i;
    int py = -7;
    int kidx = 16*(int)(r9*7);
    float out = 0;
    float num = 0;
    for (i=0;i<15;i++)
    {
        if (py+y>=0 && py+y<h)
        {
            out += output[count+py*w] * c_kernel[kidx];
            num += c_kernel[kidx];
        }
        py++;
        kidx++;
    }

    output[count] = out/num;
}

__global__ void forward_crop_layer_kernel(float *input, float *rand, int size, int c, int h, int w, int crop_height, int crop_width, int train, int flip, float angle, float *output, float scaling)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= size) return;

    float cx = w/2.;
    float cy = h/2.;

    int count = id;
    int j = id % crop_width;
    id /= crop_width;
    int i = id % crop_height;
    id /= crop_height;
    int k = id % c;
    id /= c;
    int b = id;

    float r4 = rand[16*b + 4];
    float r5 = rand[16*b + 5];
    float r6 = rand[16*b + 6];
    float r7 = rand[16*b + 7];
    float r8 = rand[16*b + 8];

    float dw = (w - crop_width)*r4;
    float dh = (h - crop_height)*r5;
    flip = (flip && (r6 > .5));
    angle = 2.*angle*r7 - angle;
    scaling = r8*2.*(scaling - 1.) + 1. - (scaling - 1.);

    if(!train){
        dw = (w - crop_width)/2.;
        dh = (h - crop_height)/2.;
        flip = 0;
        angle = 0;
        scaling = 1;
    }

    input += w*h*c*b;

    float x = (flip) ? w - dw - j - 1 : j + dw;    
    float y = i + dh;

    float rx = scaling*cos(angle)*(x-cx) - scaling*sin(angle)*(y-cy) + cx;
    float ry = scaling*sin(angle)*(x-cx) + scaling*cos(angle)*(y-cy) + cy;

    output[count] = bilinear_interpolate_kernel(input, w, h, rx, ry, k);
}




extern "C" void forward_crop_layer_gpu(crop_layer layer, network_state state)
{
    cuda_random(layer.rand_gpu, layer.batch*16);

    float radians = layer.angle*3.14159265/180.;

    float scale = 2;
    float translate = -1;
    if(layer.noadjust){
        scale = 1;
        translate = 0;
    }

    int size = layer.batch * layer.w * layer.h;

    levels_image_kernel<<<cuda_gridsize(size), BLOCK>>>(state.input, layer.rand_gpu, layer.batch, layer.c, layer.w, layer.h, state.train, layer.saturation, layer.exposure, translate, scale, layer.shift);
    check_error(cudaPeekAtLastError());

    size = layer.batch*layer.c*layer.out_w*layer.out_h;

    forward_crop_layer_kernel<<<cuda_gridsize(size), BLOCK>>>(state.input, layer.rand_gpu, size, layer.c, layer.h, layer.w, layer.out_h, layer.out_w, state.train, layer.flip, radians, layer.output_gpu, layer.scaling);
    check_error(cudaPeekAtLastError());

    if (state.train && layer.blur>0.0001)
    {
        float kernel[16*8];
        int i,j,r;
        for (j=1;j<8;j++)
        {
            int jidx = (j-1)*16;
            double segma = (double)(j)*0.3*layer.blur;
            segma *= 2.0*segma;
            r = -7;
            float sum = 0;
            for (i = 0; i < 15; i++)
            {
                kernel[i+jidx] = exp(-(r*r) / segma);
                sum += kernel[i+jidx];
                r++;
            }
            sum /= 15;
            for (i = 0; i < 15; i++)
            {
                kernel[i+jidx] /= sum;
            }
        }
        cudaMemcpyToSymbol(c_kernel, kernel, 16*8*sizeof(float));

        forward_crop_blurX_kernel<<<cuda_gridsize(size), BLOCK>>>(state.input, layer.rand_gpu, size, layer.c, layer.out_h, layer.out_w, layer.output_gpu);
        check_error(cudaPeekAtLastError());
        forward_crop_blurY_kernel<<<cuda_gridsize(size), BLOCK>>>(state.input, layer.rand_gpu, size, layer.c, layer.out_h, layer.out_w, layer.output_gpu);
        check_error(cudaPeekAtLastError());
    }

//       cuda_pull_array(layer.output_gpu, layer.output, size);
//       image im = float_to_image(layer.out_w, layer.out_h, layer.c, layer.output + 0*(layer.c*layer.out_w*layer.out_h));
//       image im2 = float_to_image(layer.out_w, layer.out_h, layer.c, layer.output + 1*(layer.c*layer.out_w*layer.out_h));
//       image im3 = float_to_image(layer.out_w, layer.out_h, layer.c, layer.output + 2*(layer.c*layer.out_w*layer.out_h));

//       translate_image(im, -translate);
//       scale_image(im, 1/scale);
//       translate_image(im2, -translate);
//       scale_image(im2, 1/scale);
//       translate_image(im3, -translate);
//       scale_image(im3, 1/scale);
       
//       show_image(im, "cropped");
//       show_image(im2, "cropped2");
//       show_image(im3, "cropped3");
//       cvWaitKey(0);


}

