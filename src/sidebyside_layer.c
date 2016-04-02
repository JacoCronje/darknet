#include "sidebyside_layer.h"
#include "cuda.h"
#include "blas.h"
#include <stdio.h>
#include <assert.h>

layer make_sidebyside_layer(int batch, int splits, int w, int h, int c)
{
    fprintf(stderr,"SideBySide Layer: Split and side by side channels in %d groups.\n", splits);
    layer l = {0};
    l.type = SIDEBYSIDE;
    l.batch = batch;
    l.w = w;
    l.h = h;
    l.c = c;
    l.out_w = w;
    l.out_h = h*splits;
    l.out_c = c/splits;
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

void forward_sidebyside_layer(const layer l, network_state state)
{
    int c, b;
    for (b=0;b<l.batch;b++)
    {
        int numInSplit = l.c / l.index;
        for (c=0;c<l.c;c++)
        {
            int sidepos = c / numInSplit;
            int newC = c % numInSplit;
            copy_cpu(l.w*l.h, state.input+b*l.inputs+l.w*l.h*c, 1,
                              l.output+b*l.outputs+newC*l.out_h*l.out_w+sidepos*l.w*l.h, 1);
        }
    }
    activate_array(l.output, l.outputs*l.batch, l.activation);
}

void backward_sidebyside_layer(const layer l, network_state state)
{
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);
    int c, b;
    for (b=0;b<l.batch;b++)
    {
        int numInSplit = l.c / l.index;
        for (c=0;c<l.c;c++)
        {
            int sidepos = c / numInSplit;
            int newC = c % numInSplit;
            axpy_cpu(l.w*l.h, 1, l.delta+b*l.outputs+newC*l.out_h*l.out_w+sidepos*l.w*l.h, 1,
                                   state.delta+b*l.inputs+l.w*l.h*c, 1);
        }
    }
}

#ifdef GPU
void forward_sidebyside_layer_gpu(const layer l, network_state state)
{
    int c, b;
    for (b=0;b<l.batch;b++)
    {
        int numInSplit = l.c / l.index;
        for (c=0;c<l.c;c++)
        {
            int sidepos = c / numInSplit;
            int newC = c % numInSplit;
            copy_ongpu(l.w*l.h, state.input+b*l.inputs+l.w*l.h*c, 1,
                                l.output_gpu+b*l.outputs+newC*l.out_h*l.out_w+sidepos*l.w*l.h, 1);
        }
    }
    activate_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation);
}

void backward_sidebyside_layer_gpu(const layer l, network_state state)
{
    gradient_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
    int c, b;
    for (b=0;b<l.batch;b++)
    {
        int numInSplit = l.c / l.index;
        for (c=0;c<l.c;c++)
        {
            int sidepos = c / numInSplit;
            int newC = c % numInSplit;
            axpy_ongpu(l.w*l.h, 1, l.delta_gpu+b*l.outputs+newC*l.out_h*l.out_w+sidepos*l.w*l.h, 1,
                                   state.delta+b*l.inputs+l.w*l.h*c, 1);
        }
    }
}
#endif
