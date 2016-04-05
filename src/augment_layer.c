#include "augment_layer.h"
#include "cuda.h"
#include "blas.h"
#include <stdio.h>
#include <assert.h>

layer make_augment_layer(int batch, int splits, int w, int h, int c)
{
    fprintf(stderr,"augment Layer: Split and side by side channels in %d groups.\n", splits);
    layer l = {0};
    l.type = AUGMENT;
    l.index = splits;
    l.batch = batch;
    if (splits==1)
    {
        l.w = w;
        l.h = h;
        l.c = c;
        l.out_w = w;
        l.out_h = h*2+10;
        l.out_c = c;
    }
    if (splits==-1)
    {
        l.w = w;
        l.h = h;
        l.c = c;
        l.out_w = w;
        l.out_h = (h-10)/2;
        l.out_c = c;
    }
    l.outputs = l.out_w * l.out_h * l.out_c;
    l.inputs = w*h*c;

    l.delta =  calloc(l.outputs*batch, sizeof(float));
    l.output = calloc(l.outputs*batch, sizeof(float));;
    #ifdef GPU
    l.delta_gpu =  cuda_make_array(l.delta, l.outputs*batch);
    l.output_gpu = cuda_make_array(l.output, l.outputs*batch);
    #endif
    return l;
}

void forward_augment_layer(const layer l, network_state state)
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

void backward_augment_layer(const layer l, network_state state)
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
void forward_augment_layer_gpu(const layer l, network_state state)
{
    int c, b;
    for (b=0;b<l.batch;b++)
    {
        if (l.index==1)
        {
            const_ongpu(l.out_h*l.out_w*l.c, 0, l.output_gpu+b*l.outputs, 1);
            for (c=0;c<l.c;c++)
            {
                copy_ongpu(l.w*l.h, state.input+b*l.inputs+l.w*l.h*c, 1,
                                    l.output_gpu+b*l.outputs+c*l.out_h*l.out_w, 1);
            }
            for (c=0;c<l.c;c++)
            {
                augmentflip_gpu(l.w, l.h, state.input+b*l.inputs+l.w*l.h*c,
                                l.output_gpu+b*l.outputs+c*l.out_h*l.out_w+l.w*(10+l.h));
            }
        } else if (l.index==-1)
        {
            for (c=0;c<l.c;c++)
            {
                copy_ongpu(l.out_h*l.out_w, state.input+b*l.inputs+l.w*l.h*c, 1,
                                    l.output_gpu+b*l.outputs+c*l.out_h*l.out_w, 1);
            }
            for (c=0;c<l.c;c++)
            {
                augmentflip_delta_gpu(l.out_w, l.out_h, 1.0, state.input+b*l.inputs+l.w*l.h*c+(10+l.out_h)*l.out_w,
                                l.output_gpu+b*l.outputs+c*l.out_h*l.out_w);
            }
        }
    }
    activate_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation);
}

void backward_augment_layer_gpu(const layer l, network_state state)
{
    gradient_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
    int c, b;
    for (b=0;b<l.batch;b++)
    {
        if (l.index==1)
        {
            for (c=0;c<l.c;c++)
            {
                axpy_ongpu(l.w*l.h, 0.5, l.delta_gpu+b*l.outputs+c*l.out_h*l.out_w, 1,
                                       state.delta+b*l.inputs+l.w*l.h*c, 1);
            }
            for (c=0;c<l.c;c++)
            {
                augmentflip_delta_gpu(l.w, l.h, 0.5, l.delta_gpu+b*l.outputs+c*l.out_h*l.out_w+l.w*(10+l.h),
                                                state.delta+b*l.inputs+l.w*l.h*c);
            }
        } else if (l.index==-1)
        {
            for (c=0;c<l.c;c++)
            {
                axpy_ongpu(l.out_h*l.out_w, 0.5, l.delta_gpu+b*l.outputs+c*l.out_h*l.out_w, 1,
                                       state.delta+b*l.inputs+l.w*l.h*c, 1);
            }
            for (c=0;c<l.c;c++)
            {
                augmentflip_delta_gpu(l.out_w, l.out_h, 0.5, l.delta_gpu+b*l.outputs+c*l.out_h*l.out_w,
                                                state.delta+b*l.inputs+l.w*l.h*c+l.out_w*(10+l.out_h));
            }
        }
    }
}
#endif
