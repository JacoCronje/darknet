#include "route_layer.h"
#include "cuda.h"
#include "blas.h"
#include <stdio.h>

route_layer make_route_layer(int batch, int n, int *input_layers, int *input_sizes, int out_w, int out_h, int out_c)
{
    fprintf(stderr,"Route Layer:");
    route_layer l = {0};
    l.type = ROUTE;
    l.batch = batch;
    l.n = n;
    l.input_layers = input_layers;
    l.input_sizes = input_sizes;
    l.out_c = out_c;
    l.out_h = out_h;
    l.out_w = out_w;
    int i;
    int outputs = out_c*out_h*out_w;
    int inputs = 0;
    for(i = 0; i < n; ++i){
        fprintf(stderr," %d", input_layers[i]);
        inputs += input_sizes[i];
    }
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

void forward_route_layer(const route_layer l, network net)
{
    int i, j;
    int offset = 0;
    for(i = 0; i < l.n; ++i){
        int index = l.input_layers[i];
        float *input = net.layers[index].output;
        if (net.layers[index].out_w == l.out_w && net.layers[index].out_h == l.out_h)
        {
            // same size, no scaling required
            int input_size = l.input_sizes[i];
            for(j = 0; j < l.batch; ++j){
                copy_cpu(input_size, input + j*input_size, 1, l.output + offset + j*l.outputs, 1);
            }
            offset += input_size;

        } else
        {
            //TODO : CPU
            // different size, scale the input
            int input_size = l.input_sizes[i];
            for(j = 0; j < l.batch; ++j){
               // copy_cpu(input_size, input + j*input_size, 1, l.output + offset + j*l.outputs, 1);
            }
            offset += l.out_h*l.out_w*net.layers[index].out_c;
        }
    }
}

void backward_route_layer(const route_layer l, network net)
{
    int i, j;
    int offset = 0;
    for(i = 0; i < l.n; ++i){
        int index = l.input_layers[i];
        float *delta = net.layers[index].delta;
        if (net.layers[index].out_w == l.out_w && net.layers[index].out_h == l.out_h)
        {
            // same size, no scaling required
            int input_size = l.input_sizes[i];
            for(j = 0; j < l.batch; ++j){
                axpy_cpu(input_size, 1, l.delta + offset + j*l.outputs, 1, delta + j*input_size, 1);
            }
            offset += input_size;
        } else
        {
            //TODO : CPU
            // different size, scale the delta
            int input_size = l.input_sizes[i];
            for(j = 0; j < l.batch; ++j){
               // axpy_cpu(input_size, 1, l.delta + offset + j*l.outputs, 1, delta + j*input_size, 1);
            }
            offset += l.out_h*l.out_w*net.layers[index].out_c;
        }
    }
}

#ifdef GPU
void forward_route_layer_gpu(const route_layer l, network net)
{
    int i, j;
    int offset = 0;
    for(i = 0; i < l.n; ++i){
        int index = l.input_layers[i];
        float *input = net.layers[index].output_gpu;
        if (net.layers[index].out_w == l.out_w && net.layers[index].out_h == l.out_h)
        {
            // same size, no scaling required
            int input_size = l.input_sizes[i];
            for(j = 0; j < l.batch; ++j){
                copy_ongpu(input_size, input + j*input_size, 1, l.output_gpu + offset + j*l.outputs, 1);
            }
            offset += input_size;
        } else
        {
            // different size, scale the input
            int input_size = l.input_sizes[i];
            for(j = 0; j < l.batch; ++j){
                routescale_gpu(net.layers[index].out_w, net.layers[index].out_h, net.layers[index].out_c,
                                  input + j*input_size,
                                  l.out_w, l.out_h, l.out_c, l.output_gpu + offset + j*l.outputs);
            }
            offset += l.out_h*l.out_w*net.layers[index].out_c;
        }
    }
}

void backward_route_layer_gpu(const route_layer l, network net)
{
    int i, j;
    int offset = 0;
    for(i = 0; i < l.n; ++i){
        int index = l.input_layers[i];
        float *delta = net.layers[index].delta_gpu;
        if (net.layers[index].out_w == l.out_w && net.layers[index].out_h == l.out_h)
        {
            // same size, no scaling required
            int input_size = l.input_sizes[i];
            for(j = 0; j < l.batch; ++j){
                axpy_ongpu(input_size, 1, l.delta_gpu + offset + j*l.outputs, 1, delta + j*input_size, 1);
            }
            offset += input_size;
        } else
        {
            // different size, scale the delta
            int input_size = l.input_sizes[i];
            for(j = 0; j < l.batch; ++j){
                routedelta_gpu(net.layers[index].out_w, net.layers[index].out_h, net.layers[index].out_c,
                                  delta + j*input_size,
                                  l.out_w, l.out_h, l.out_c, l.delta_gpu + offset + j*l.outputs);
            }
            offset += l.out_h*l.out_w*net.layers[index].out_c;
        }
    }
}
#endif
