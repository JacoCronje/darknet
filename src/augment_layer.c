#include "augment_layer.h"
#include "cuda.h"
#include "blas.h"
#include <stdio.h>
#include <assert.h>

layer make_augment_layer(int batch, int splits, int gap, int n_angles, int* angles, int w, int h, int c)
{
    int i;
    layer l = {0};
    l.type = AUGMENT;
    l.index = splits;
    l.gap = gap;
    l.batch = batch;
    l.w = w;
    l.h = h;
    l.c = c;
    if (splits==1) // Flip
    {
        fprintf(stderr,"augment Layer: Flip. Gap = %d\n", gap, splits);
        l.out_w = w;
        l.out_h = h*2+l.gap;
        l.out_c = c;
    }
    else if (splits==-1) // Flip
    {
        fprintf(stderr,"augment Layer: Merge Flip. Gap = %d\n", gap, splits);
        l.out_w = w;
        l.out_h = (h-l.gap)/2;
        l.out_c = c;
    }
    else if (splits==2) // Rotation
    {
        fprintf(stderr,"augment Layer: Rotate. Gap = %d Angles = %d", gap, splits,angles[0]);
        for (i=1;i<n_angles;i++)
            fprintf(stderr, ",%d", angles[i]);
        fprintf(stderr, "\n");
        l.n_angles = n_angles;
        l.angles = angles;
        l.out_w = w;
        l.out_h = h*(1+n_angles) + l.gap*n_angles;
        l.out_c = c;
    }
    else
    {
        error("Invalid augmentation type.");
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
    for (b=0;b<l.batch;b++)
    {
        if (l.index==1)
        {
            // flip
            const_ongpu(l.out_h*l.out_w*l.c, 0, l.output_gpu+b*l.outputs, 1);
            for (c=0;c<l.c;c++)
            {
                copy_ongpu(l.w*l.h, state.input+b*l.inputs+l.w*l.h*c, 1,
                                    l.output_gpu+b*l.outputs+c*l.out_h*l.out_w, 1);
            }
            for (c=0;c<l.c;c++)
            {
                augmentflip_gpu(l.w, l.h, state.input+b*l.inputs+l.w*l.h*c,
                                l.output_gpu+b*l.outputs+c*l.out_h*l.out_w+l.w*(l.gap+l.h));
            }
        } else if (l.index==2)
        {
            // rotate
            const_ongpu(l.out_h*l.out_w*l.c, 0, l.output_gpu+b*l.outputs, 1);
            for (c=0;c<l.c;c++)
            {
                copy_ongpu(l.w*l.h, state.input+b*l.inputs+l.w*l.h*c, 1,
                                    l.output_gpu+b*l.outputs+c*l.out_h*l.out_w, 1);
            }
            for (a=0;a<l.n_angles;a++)
            for (c=0;c<l.c;c++)
            {
                augmentrotate_gpu(l.w, l.h, state.input+b*l.inputs+l.w*l.h*c,
                                  l.output_gpu+b*l.outputs+c*l.out_h*l.out_w+l.w*(l.gap+l.h)*(1+a), l.angles[a]);
            }
        } else if (l.index==-1)
        {
            // flip
            for (c=0;c<l.c;c++)
            {
                copy_ongpu(l.out_h*l.out_w, state.input+b*l.inputs+l.w*l.h*c, 1,
                                    l.output_gpu+b*l.outputs+c*l.out_h*l.out_w, 1);
            }
            for (c=0;c<l.c;c++)
            {
                augmentflip_delta_gpu(l.out_w, l.out_h, 1.0, state.input+b*l.inputs+l.w*l.h*c+(l.gap+l.out_h)*l.out_w,
                                l.output_gpu+b*l.outputs+c*l.out_h*l.out_w);
            }
        }
    }
    activate_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation);
}

void backward_augment_layer_gpu(const layer l, network_state state)
{
    gradient_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
    int c, b, a;
    for (b=0;b<l.batch;b++)
    {
        if (l.index==1)
        {
            // flip
            for (c=0;c<l.c;c++)
            {
                axpy_ongpu(l.w*l.h, 0.5, l.delta_gpu+b*l.outputs+c*l.out_h*l.out_w, 1,
                                       state.delta+b*l.inputs+l.w*l.h*c, 1);
            }
            for (c=0;c<l.c;c++)
            {
                augmentflip_delta_gpu(l.w, l.h, 0.5, l.delta_gpu+b*l.outputs+c*l.out_h*l.out_w+l.w*(l.gap+l.h),
                                                state.delta+b*l.inputs+l.w*l.h*c);
            }
        } else if (l.index==2)
        {
            // rotate
            float factor = 1.0 / (1+l.n_angles);
            for (c=0;c<l.c;c++)
            {
                axpy_ongpu(l.w*l.h, factor, l.delta_gpu+b*l.outputs+c*l.out_h*l.out_w, 1,
                                       state.delta+b*l.inputs+l.w*l.h*c, 1);
            }
            for (a=0;a<l.n_angles;a++)
            for (c=0;c<l.c;c++)
            {
                augmentrotate_delta_gpu(l.w, l.h, factor,
                                        l.delta_gpu+b*l.outputs+c*l.out_h*l.out_w+l.w*(l.gap+l.h)*(1+a),
                                        state.delta+b*l.inputs+l.w*l.h*c, l.angles[a]);
            }
        } else if (l.index==-1)
        {
            // flip
            for (c=0;c<l.c;c++)
            {
                axpy_ongpu(l.out_h*l.out_w, 0.5, l.delta_gpu+b*l.outputs+c*l.out_h*l.out_w, 1,
                                       state.delta+b*l.inputs+l.w*l.h*c, 1);
            }
            for (c=0;c<l.c;c++)
            {
                augmentflip_delta_gpu(l.out_w, l.out_h, 0.5, l.delta_gpu+b*l.outputs+c*l.out_h*l.out_w,
                                                state.delta+b*l.inputs+l.w*l.h*c+l.out_w*(l.gap+l.out_h));
            }
        }
    }
}
#endif
