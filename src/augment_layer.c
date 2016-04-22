#include "augment_layer.h"
#include "cuda.h"
#include "blas.h"
#include "utils.h"
#include <stdio.h>
#include <assert.h>

/*
  TODO:
  - random options
  - try pixel lookup instead of bilinear
  */

layer make_augment_layer(int batch, int merge, int gap, int n_aug, float* angles, int *flips, float* scales, int w, int h, int c)
{
    int i;
    layer l = {0};
    l.type = AUGMENT;
    l.index = merge;
    l.gap = gap;
    l.n_aug = n_aug;
    l.batch = batch;
    l.w = w;
    l.h = h;
    l.c = c;

    fprintf(stderr,"Augment Layer: Gap=%d Ang=%.2f", gap, angles[0]);
    for (i=1;i<n_aug;i++)
        fprintf(stderr, ",%.2f", angles[i]);
    fprintf(stderr," F=%d", flips[0]);
    for (i=1;i<n_aug;i++)
        fprintf(stderr, ",%d", flips[i]);
    fprintf(stderr," S=%.2f", scales[0]);
    for (i=1;i<n_aug;i++)
        fprintf(stderr, ",%.2f", scales[i]);

    l.angles = angles;
    l.flips = flips;
    l.scales = scales;
    l.out_w = w;
    if (merge==2)
    {
        l.out_h = (h-l.gap*n_aug)/(1+n_aug);
        l.out_c = c*(1+n_aug);
        fprintf(stderr, "[merge_split]");
    } else if (merge==1)
    {
        l.out_h = (h-l.gap*n_aug)/(1+n_aug);
        l.out_c = c;
        fprintf(stderr, "[merge_max]");
    }
    else if (merge==0)
    {
        l.out_h = h*(1+n_aug) + l.gap*n_aug;
        l.out_c = c;
    }
    fprintf(stderr, "\n               ");

    fprintf(stderr, ": %d x %d x %d image, -> %d x %d x %d image\n", l.h,l.w,l.c, l.out_h, l.out_w, l.out_c);

    l.outputs = l.out_w * l.out_h * l.out_c;
    l.inputs = w*h*c;

    l.delta =  calloc(l.outputs*batch, sizeof(float));
    l.output = calloc(l.outputs*batch, sizeof(float));;
    #ifdef GPU
    l.delta_gpu =  cuda_make_array(l.delta, l.outputs*batch);
    l.output_gpu = cuda_make_array(l.output, l.outputs*batch);
    #endif
    if (merge==1 || merge==2)
    {
        l.indexes = calloc(l.outputs*batch, sizeof(int));
        #ifdef GPU
        l.indexes_gpu = cuda_make_int_array(l.outputs*batch);
        #endif
    }
    return l;
}

void forward_augment_layer(const layer l, network_state state)
{
    //TODO
}

void backward_augment_layer(const layer l, network_state state)
{
    //TODO
}

#ifdef GPU
void forward_augment_layer_gpu(const layer l, network_state state)
{
    int c, b, a;

    if (l.index==0)
    {
        // clear
        const_ongpu(l.out_h*l.out_w*l.c*l.batch, 0, l.output_gpu+b*l.outputs, 1);
        for (b=0;b<l.batch;b++)
        {
            // original
            for (c=0;c<l.c;c++)
            {
                copy_ongpu(l.w*l.h, state.input+b*l.inputs+l.w*l.h*c, 1,
                                    l.output_gpu+b*l.outputs+c*l.out_h*l.out_w, 1);
            }
            for (a=0;a<l.n_aug;a++)
            {
                for (c=0;c<l.c;c++)
                {
                    augment_forward_gpu(l.w, l.h, state.input+b*l.inputs+l.w*l.h*c,
                                        l.output_gpu+b*l.outputs+c*l.out_h*l.out_w+l.w*(l.gap+l.h)*(1+a),
                                        l.angles[a], l.flips[a], l.scales[a]);
                }
            }
        }
    } else if (l.index==1)
    {
        // merging with max
        for (b=0;b<l.batch;b++)
        {
            augment_forward_max_gpu(l.w, l.h, l.c, l.out_w, l.out_h, l.gap,
                                    state.input+b*l.inputs,
                                    l.output_gpu+b*l.outputs,
                                    l.indexes_gpu,
                                    l.n_aug,
                                    l.angles, l.flips, l.scales);
        }
    } else if (l.index==2)
    {
        // merging with split, channels increased
        for (b=0;b<l.batch;b++)
        {
            augment_forward_split_gpu(l.w, l.h, l.c, l.out_w, l.out_h, l.gap,
                                    state.input+b*l.inputs,
                                    l.output_gpu+b*l.outputs,
                                    l.indexes_gpu,
                                    l.n_aug,
                                    l.angles, l.flips, l.scales);
        }
    }
    activate_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation);
}

void backward_augment_layer_gpu(const layer l, network_state state)
{
    gradient_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
    if (!state.delta) return;
    int c, b, a;
    if (l.index==0)
    {
        for (b=0;b<l.batch;b++)
        {
            // original
            for (c=0;c<l.c;c++)
            {
                axpy_ongpu(l.w*l.h, 1.0, l.delta_gpu+b*l.outputs+c*l.out_h*l.out_w, 1,
                                         state.delta+b*l.inputs+l.w*l.h*c, 1);
            }
            for (a=0;a<l.n_aug;a++)
            for (c=0;c<l.c;c++)
            {
                augment_backward_gpu(l.w, l.h, 1.0,
                                     l.delta_gpu+b*l.outputs+c*l.out_h*l.out_w+l.w*(l.gap+l.h)*(1+a),
                                     state.delta+b*l.inputs+l.w*l.h*c, l.angles[a], l.flips[a], l.scales[a]);
            }
        }
    } else if (l.index==1)
    {
        // merging with max
        for (b=0;b<l.batch;b++)
        {
            augment_backward_max_gpu(l.w, l.h, l.c, l.out_w, l.out_h, l.gap,
                                    l.delta_gpu+b*l.outputs,
                                    state.delta+b*l.inputs,
                                    l.indexes_gpu);
        }
    } else if (l.index==2)
    {
        // merging with split
        for (b=0;b<l.batch;b++)
        {
            augment_backward_split_gpu(l.out_w, l.out_h, l.out_c,
                                    l.delta_gpu+b*l.outputs,
                                    state.delta+b*l.inputs,
                                    l.indexes_gpu);
        }
    }
}
#endif
