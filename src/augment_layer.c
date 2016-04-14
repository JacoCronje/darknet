#include "augment_layer.h"
#include "cuda.h"
#include "blas.h"
#include "utils.h"
#include <stdio.h>
#include <assert.h>

/*
  TODO:
  - merge layer
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

    fprintf(stderr,"Augment Layer: Gap=%d Angles=%.2f", gap, angles[0]);
    for (i=1;i<n_aug;i++)
        fprintf(stderr, ",%.2f", angles[i]);
    fprintf(stderr," Flips=%d", flips[0]);
    for (i=1;i<n_aug;i++)
        fprintf(stderr, ",%d", flips[i]);
    fprintf(stderr," Scales=%.2f", scales[0]);
    for (i=1;i<n_aug;i++)
        fprintf(stderr, ",%.2f", scales[i]);

    l.angles = angles;
    l.flips = flips;
    l.scales = scales;
    l.out_w = w;
    if (merge==1)
    {
        l.out_h = (h-l.gap*n_aug)/(1+n_aug);
        fprintf(stderr, "[merge]");
    }
    else
    {
        l.out_h = h*(1+n_aug) + l.gap*n_aug;
    }
    l.out_c = c;
    fprintf(stderr, "\n               ");

//    if (splits==1) // Flip
//    {
//        fprintf(stderr,"augment Layer: Flip. Gap = %d", gap);
//        l.out_w = w;
//        l.out_h = h*2+l.gap;
//        l.out_c = c;
//    }
//    else if (splits==-1) // Flip
//    {
//        fprintf(stderr,"augment Layer: Merge Flip. Gap = %d", gap);
//        l.out_w = w;
//        l.out_h = (h-l.gap)/2;
//        l.out_c = c;
//    }
//    else if (splits==2) // Rotation
//    {
//        fprintf(stderr,"augment Layer: Rotate. Gap = %d Angles = %d", gap, angles[0]);
//        for (i=1;i<n_angles;i++)
//            fprintf(stderr, ",%d", angles[i]);
//        l.n_angles = n_angles;
//        l.angles = angles;
//        l.out_w = w;
//        l.out_h = h*(1+n_angles) + l.gap*n_angles;
//        l.out_c = c;
//    }
//    else
//    {
//        error("Invalid augmentation type.");
//    }
    fprintf(stderr, ": %d x %d x %d image, -> %d x %d x %d image\n", l.h,l.w,l.c, l.out_h, l.out_w, l.out_c);

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
            for (c=0;c<l.c;c++)
            {
                augment_forward_gpu(l.w, l.h, state.input+b*l.inputs+l.w*l.h*c,
                                    l.output_gpu+b*l.outputs+c*l.out_h*l.out_w+l.w*(l.gap+l.h)*(1+a),
                                    l.angles[a], l.flips[a], l.scales[a]);
            }
        }

        // augment_flip_gpu(l.w, l.h, l.c, l.batch, l.gap, state.input, l.output_gpu);
    }

//    for (b=0;b<l.batch;b++)
//    {
//        if (l.index==1)
//        {
////            // flip
////            const_ongpu(l.out_h*l.out_w*l.c, 0, l.output_gpu+b*l.outputs, 1);
////            for (c=0;c<l.c;c++)
////            {
////                copy_ongpu(l.w*l.h, state.input+b*l.inputs+l.w*l.h*c, 1,
////                                    l.output_gpu+b*l.outputs+c*l.out_h*l.out_w, 1);
////            }
////            for (c=0;c<l.c;c++)
////            {
////                augmentflip_gpu(l.w, l.h, state.input+b*l.inputs+l.w*l.h*c,
////                                l.output_gpu+b*l.outputs+c*l.out_h*l.out_w+l.w*(l.gap+l.h));
////            }
//        } else if (l.index==2)
//        {
//            // rotate
//            const_ongpu(l.out_h*l.out_w*l.c, 0, l.output_gpu+b*l.outputs, 1);
//            for (c=0;c<l.c;c++)
//            {
//                copy_ongpu(l.w*l.h, state.input+b*l.inputs+l.w*l.h*c, 1,
//                                    l.output_gpu+b*l.outputs+c*l.out_h*l.out_w, 1);
//            }
//            for (a=0;a<l.n_angles;a++)
//            for (c=0;c<l.c;c++)
//            {
//                augmentrotate_gpu(l.w, l.h, state.input+b*l.inputs+l.w*l.h*c,
//                                  l.output_gpu+b*l.outputs+c*l.out_h*l.out_w+l.w*(l.gap+l.h)*(1+a), l.angles[a]);
//            }
//        } else if (l.index==-1)
//        {
//            // flip
//            for (c=0;c<l.c;c++)
//            {
//                copy_ongpu(l.out_h*l.out_w, state.input+b*l.inputs+l.w*l.h*c, 1,
//                                    l.output_gpu+b*l.outputs+c*l.out_h*l.out_w, 1);
//            }
//            for (c=0;c<l.c;c++)
//            {
//                augmentflip_delta_gpu(l.out_w, l.out_h, 1.0, state.input+b*l.inputs+l.w*l.h*c+(l.gap+l.out_h)*l.out_w,
//                                l.output_gpu+b*l.outputs+c*l.out_h*l.out_w);
//            }
//        }
//    }
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
    }

//    if (l.index==1)
//    {
//        augment_flip_delta_gpu(l.w, l.h, l.c, l.batch, l.gap, l.delta_gpu, state.delta);
//    }


//    int c, b, a;
//    for (b=0;b<l.batch;b++)
//    {
//        if (l.index==1)
//        {
//            // flip
////            for (c=0;c<l.c;c++)
////            {
////                axpy_ongpu(l.w*l.h, 0.5, l.delta_gpu+b*l.outputs+c*l.out_h*l.out_w, 1,
////                                       state.delta+b*l.inputs+l.w*l.h*c, 1);
////            }
////            for (c=0;c<l.c;c++)
////            {
////                augmentflip_delta_gpu(l.w, l.h, 0.5, l.delta_gpu+b*l.outputs+c*l.out_h*l.out_w+l.w*(l.gap+l.h),
////                                                state.delta+b*l.inputs+l.w*l.h*c);
////            }
//        } else if (l.index==2)
//        {
//            // rotate
//            float factor = 1.0;// / (1+l.n_angles);
//            for (c=0;c<l.c;c++)
//            {
//                axpy_ongpu(l.w*l.h, factor, l.delta_gpu+b*l.outputs+c*l.out_h*l.out_w, 1,
//                                       state.delta+b*l.inputs+l.w*l.h*c, 1);
//            }
//            for (a=0;a<l.n_angles;a++)
//            for (c=0;c<l.c;c++)
//            {
//                augmentrotate_delta_gpu(l.w, l.h, factor,
//                                        l.delta_gpu+b*l.outputs+c*l.out_h*l.out_w+l.w*(l.gap+l.h)*(1+a),
//                                        state.delta+b*l.inputs+l.w*l.h*c, l.angles[a]);
//            }
//        } else if (l.index==-1)
//        {
//            // flip
//            for (c=0;c<l.c;c++)
//            {
//                axpy_ongpu(l.out_h*l.out_w, 0.5, l.delta_gpu+b*l.outputs+c*l.out_h*l.out_w, 1,
//                                       state.delta+b*l.inputs+l.w*l.h*c, 1);
//            }
//            for (c=0;c<l.c;c++)
//            {
//                augmentflip_delta_gpu(l.out_w, l.out_h, 0.5, l.delta_gpu+b*l.outputs+c*l.out_h*l.out_w,
//                                                state.delta+b*l.inputs+l.w*l.h*c+l.out_w*(l.gap+l.out_h));
//            }
//        }
//    }
}
#endif
