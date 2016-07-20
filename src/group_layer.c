#include "group_layer.h"
#include "cuda.h"
#include "blas.h"
#include <stdio.h>

group_layer make_group_layer(int batch, int n, int *output_channels, int w, int h, int c)
{
    fprintf(stderr,"Group Layer:");
    group_layer l = {0};
    l.type = GROUP;
    l.batch = batch;
    l.n = n;
    l.input_layers = output_channels;
    l.out_c = n;
    l.out_h = h;
    l.out_w = w;
    l.w = w;
    l.h = h;
    l.c = c;

    int i;
    int outputs = l.out_c*l.out_h*l.out_w;
    int inputs = c*w*h;
    for(i = 0; i < n; ++i){
        fprintf(stderr," %d", output_channels[i]);
    }
    fprintf(stderr, ": %d x %d x %d image, -> %d x %d x %d image\n", l.h,l.w,l.c, l.out_h, l.out_w, l.out_c);
    fprintf(stderr, "\n");
    l.outputs = outputs;
    l.inputs = inputs;
    l.delta =  calloc(outputs*batch, sizeof(float));
    l.output = calloc(outputs*batch, sizeof(float));
    #ifdef GPU
    l.delta_gpu =  cuda_make_array(l.delta, outputs*batch);
    l.output_gpu = cuda_make_array(l.output, outputs*batch);
    #endif
    return l;
}

void forward_group_layer(const group_layer l, network_state state)
{
    // TODO:CPU
    fprintf(stderr, "CPU CODE CALLED!\n");
}

void backward_group_layer(const group_layer l, network_state state)
{
    fprintf(stderr, "CPU CODE CALLED!\n");
    // TODO:CPU
}


#ifdef GPU
void forward_group_layer_gpu(const group_layer l, network net)
{
    int b, c;
    for (b=0;b<l.batch;b++)
    {
        for (c=0;c<l.n;c++)
        {
            //copy_ongpu(l.w*l.h, net.layers[0].output_gpu+b*l.inputs+l.input_layers[c]*l.w*l.h, 1, l.output_gpu+b*l.outputs+c*l.out_h*l.out_w, 1);
            copy_ongpu(l.w*l.h, l.net_input+b*12*l.w*l.h+l.input_layers[c]*l.w*l.h, 1, l.output_gpu+b*l.outputs+c*l.out_h*l.out_w, 1);

        }
    }
}

void backward_group_layer_gpu(const group_layer l, network net)
{
//    int b, c;
//    if (l.delta)
//    {
//        for (b=0;b<l.batch;b++)
//        {
//            for (c=0;c<l.n;c++)
//            {
//                axpy_ongpu(l.w*l.h, 1, l.delta_gpu+b*l.outputs+c*l.out_w*l.out_h, 1, state.delta+b*l.inputs+l.input_layers[c]*l.w*l.h, 1);
//            }
//        }
//    }
}
#endif
