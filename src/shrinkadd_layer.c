#include "shrinkadd_layer.h"
#include "cuda.h"
#include "blas.h"
#include <stdio.h>
#include <assert.h>

layer make_shrinkadd_layer(int batch, int splits, int w, int h, int c)
{
    fprintf(stderr,"shrinkadd Layer: Shrink height by adding feature map with %d groups.\n", splits);
    layer l = {0};
    l.type = SHRINKADD;
    l.batch = batch;
    l.w = w;
    l.h = h;
    l.c = c;
    l.out_w = w;
    l.out_h = h/splits;
    l.out_c = c;
    l.outputs = l.out_w * l.out_h * l.out_c;
    l.inputs = w*h*c;

    l.index = splits;

    l.delta =  calloc(l.outputs*batch, sizeof(float));
    l.output = calloc(l.outputs*batch, sizeof(float));;
    #ifdef GPU
    l.delta_gpu =  cuda_make_array(l.delta, l.outputs*batch);
    l.output_gpu = cuda_make_array(l.output, l.outputs*batch);
    #endif
    return l;
}

void forward_shrinkadd_layer(const layer l, network_state state)
{
    int c, b, i;
    for (b=0;b<l.batch;b++)
    {
        for (c=0;c<l.c;c++)
        {
            copy_cpu(l.out_w*l.out_h, state.input+b*l.inputs+l.w*l.h*c, 1,
                                l.output+b*l.outputs+c*l.out_h*l.out_w, 1);
            for (i=1;i<l.index;i++)
            {
            axpy_cpu(l.out_w*l.out_h, 1, state.input+b*l.inputs+l.w*l.h*c+i*l.out_w*l.out_h, 1,
                                           l.output+b*l.outputs+c*l.out_h*l.out_w, 1);
            }
        }
    }
    activate_array(l.output, l.outputs*l.batch, l.activation);
}

void backward_shrinkadd_layer(const layer l, network_state state)
{

    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);
    int c, b, i;
    for (b=0;b<l.batch;b++)
    {
        for (c=0;c<l.c;c++)
        {
            for (i=0;i<l.index;i++)
            {
            axpy_cpu(l.out_w*l.out_h, 1, l.delta+b*l.outputs+c*l.out_h*l.out_w, 1,
                                         state.delta+b*l.inputs+l.w*l.h*c+i*l.out_w*l.out_h, 1);
            }
        }
    }
}

#ifdef GPU
void forward_shrinkadd_layer_gpu(const layer l, network_state state)
{
    int c, b, i;
    for (b=0;b<l.batch;b++)
    {
        for (c=0;c<l.c;c++)
        {
            copy_ongpu(l.out_w*l.out_h, state.input+b*l.inputs+l.w*l.h*c, 1,
                                l.output_gpu+b*l.outputs+c*l.out_h*l.out_w, 1);
            for (i=1;i<l.index;i++)
            {
            axpy_ongpu(l.out_w*l.out_h, 1, state.input+b*l.inputs+l.w*l.h*c+i*l.out_w*l.out_h, 1,
                                           l.output_gpu+b*l.outputs+c*l.out_h*l.out_w, 1);
            }
        }
    }
    activate_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation);
}

void backward_shrinkadd_layer_gpu(const layer l, network_state state)
{
    gradient_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
    int c, b, i;
    for (b=0;b<l.batch;b++)
    {
        for (c=0;c<l.c;c++)
        {
            for (i=0;i<l.index;i++)
            {
            axpy_ongpu(l.out_w*l.out_h, 1, l.delta_gpu+b*l.outputs+c*l.out_h*l.out_w, 1,
                                           state.delta+b*l.inputs+l.w*l.h*c+i*l.out_w*l.out_h, 1);
            }
        }
    }
}
#endif
