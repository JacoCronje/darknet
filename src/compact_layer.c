#include "compact_layer.h"
#include "cuda.h"
#include "blas.h"
#include <stdio.h>
#include <assert.h>

layer make_compact_layer(int batch, int splits, int method, int w, int h, int c)
{
    fprintf(stderr,"Compact Layer: Split and merge channels in %d groups. %d %d %d\n", splits,w ,h ,c);
    layer l = {0};
    l.type = COMPACT;
    l.batch = batch;
    l.w = w;
    l.h = h;
    l.c = c;
    l.out_w = w;
    l.out_h = h;
    if (method<10)
    {
        l.out_c = c/splits;
    } else
    {
        l.out_c = c;
    }
    l.outputs = l.out_w * l.out_h * l.out_c;
    l.inputs = w*h*c;

    l.index = splits;
    l.method = method;

    l.delta =  calloc(l.outputs*batch, sizeof(float));
    l.output = calloc(l.outputs*batch, sizeof(float));;
    l.indexes = calloc(l.outputs*batch, sizeof(int));
    #ifdef GPU
    l.delta_gpu =  cuda_make_array(l.delta, l.outputs*batch);
    l.output_gpu = cuda_make_array(l.output, l.outputs*batch);
    l.indexes_gpu = cuda_make_int_array(l.outputs*batch);
    #endif

    return l;
}

void forward_compact_layer(const layer l, network_state state)
{
    int i, b;
    for (b=0;b<l.batch;b++)
    {
        // copy first section
        copy_cpu(l.outputs, state.input+b*l.inputs, 1, l.output+b*l.outputs, 1);
        // add other splits
        for (i=1;i<l.index;i++)
        {
            axpy_cpu(l.outputs, 1, state.input+b*l.inputs+i*l.outputs, 1, l.output+b*l.outputs, 1);
        }
    }
    activate_array(l.output, l.outputs*l.batch, l.activation);
}

void backward_compact_layer(const layer l, network_state state)
{
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);
    int i, b;
    for (b=0;b<l.batch;b++)
    {
        for (i=0;i<l.index;i++)
        {
            axpy_cpu(l.outputs, 1, l.delta+b*l.outputs, 1, state.delta+b*l.inputs+i*l.outputs, 1);
        }
    }
}

#ifdef GPU
void forward_compact_layer_gpu(const layer l, network_state state)
{
    int i, b;
    for (b=0;b<l.batch;b++)
    {
        if (l.method==0) // add
        {
            // copy first section
            copy_ongpu(l.outputs, state.input+b*l.inputs, 1, l.output_gpu+b*l.outputs, 1);
            // add other splits
            for (i=1;i<l.index;i++)
            {
                axpy_ongpu(l.outputs, 1, state.input+b*l.inputs+i*l.outputs, 1, l.output_gpu+b*l.outputs, 1);
            }
        } else if (l.method==1) // sub
        {
            // copy first section
            copy_ongpu(l.outputs, state.input+b*l.inputs, 1, l.output_gpu+b*l.outputs, 1);
            // sub other splits
            for (i=1;i<l.index;i++)
            {
                axpy_ongpu(l.outputs, -1, state.input+b*l.inputs+i*l.outputs, 1, l.output_gpu+b*l.outputs, 1);
            }
        } else if (l.method==2) // max
        {
            compact_forward_max_gpu(l.w, l.h, l.c, l.index,
                                    state.input+b*l.inputs,
                                    l.output_gpu+b*l.outputs,
                                    l.indexes_gpu);
        } else if (l.method==10)
        {
            compact_forward_padd_gpu(l.w, l.h, l.c,
                                     state.input+b*l.inputs,
                                     l.output_gpu+b*l.outputs);
        } else if (l.method==12)
        {
            compact_forward_pmax_gpu(l.w, l.h, l.c,
                                     state.input+b*l.inputs,
                                     l.output_gpu+b*l.outputs,
                                     l.indexes_gpu);
        }
    }
    activate_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation);
}

void backward_compact_layer_gpu(const layer l, network_state state)
{
    gradient_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
    int i, b;
    for (b=0;b<l.batch;b++)
    {
        if (l.method==0) // add
        {
            for (i=0;i<l.index;i++)
            {
                axpy_ongpu(l.outputs, 1, l.delta_gpu+b*l.outputs, 1, state.delta+b*l.inputs+i*l.outputs, 1);
            }
        } else if (l.method==1) // sub
        {
            axpy_ongpu(l.outputs, 1, l.delta_gpu+b*l.outputs, 1, state.delta+b*l.inputs, 1);
            for (i=1;i<l.index;i++)
            {
                axpy_ongpu(l.outputs, -1, l.delta_gpu+b*l.outputs, 1, state.delta+b*l.inputs+i*l.outputs, 1);
            }
        } else if (l.method==2) // max
        {
            compact_backward_max_gpu(l.w, l.h, l.c, l.index,
                                     l.delta_gpu+b*l.outputs,
                                     state.delta+b*l.inputs,
                                     l.indexes_gpu);
        } else if (l.method==10)
        {
            compact_backward_padd_gpu(l.w, l.h, l.c,
                                      l.delta_gpu+b*l.outputs,
                                      state.delta+b*l.inputs);
        } else if (l.method==12)
        {
            compact_backward_pmax_gpu(l.w, l.h, l.c,
                                      l.delta_gpu+b*l.outputs,
                                      state.delta+b*l.inputs,
                                      l.indexes_gpu);
        }
    }
}
#endif
