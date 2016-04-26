#include "keypoint_layer.h"
#include "network.h"
#include "parser.h"
#include "cuda.h"
#include "blas.h"
#include "utils.h"
#include <stdio.h>
#include <assert.h>

network kp_net;

layer make_keypoint_layer(int batch, int w, int h, int c, char *cfgfile, char* weightfile)
{
    kp_net = parse_network_cfg(cfgfile);
    load_weights(&kp_net, weightfile);
    set_batch_network(&kp_net, batch);
    int i;
    layer l = {0};
    l.type = KEYPOINT;
    l.batch = batch;
    l.w = w;
    l.h = h;
    l.c = c;
    l.out_w = w;
    l.out_h = h;
    l.out_c = c+4;

    fprintf(stderr, "Keypoint : %d x %d x %d image, -> %d x %d x %d image\n", l.h,l.w,l.c, l.out_h, l.out_w, l.out_c);

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

void forward_keypoint_layer(const layer l, network_state state)
{
    //TODO
}

void backward_keypoint_layer(const layer l, network_state state)
{
    //TODO
}

#ifdef GPU
void forward_keypoint_layer_gpu(const layer l, network_state state)
{
    int c, b, a;

    // detect points
    float *p = network_predict_gpu_fromgpu(kp_net, state.input);

    for (b=0;b<l.batch;b++)
    {
        // copy original
        copy_ongpu(l.w*l.h*l.c, state.input+b*l.inputs, 1, l.output_gpu+b*l.outputs, 1);

        // draw keypoint maps
        for (c=0;c<4;c++)
        {
            keypoint_gpu(l.out_w,l.out_h, p[b*8+c*2], p[b*8+c*2+1], l.output_gpu+b*l.outputs+(l.c+c)*l.out_h*l.out_w);
        }
    }
}

void backward_keypoint_layer_gpu(const layer l, network_state state)
{
}
#endif
