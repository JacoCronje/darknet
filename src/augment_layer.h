#ifndef augment_LAYER_H
#define augment_LAYER_H

#include "layer.h"
#include "network.h"

layer make_augment_layer(int batch, int merge, int gap, int n_aug, float* angles, int *flips, float* scales, int* similar, int w, int h, int c);
void forward_augment_layer(const layer l, network_state state);
void backward_augment_layer(const layer l, network_state state);

#ifdef GPU
void forward_augment_layer_gpu(const layer l, network_state state);
void backward_augment_layer_gpu(const layer l, network_state state);

void augment_flip_gpu(int w, int h, int c, int batch, int gap, float *src, float *dest);
void augment_flip_delta_gpu(int w, int h, int c, int batch, int gap, float *src, float *dest);

void augmentflip_gpu(int w, int h, float *src, float *dest);
void augmentflip_delta_gpu(int w, int h, float ALPHA, float *src, float *dest);
void augmentrotate_gpu(int w, int h, float *src, float *dest, int ang);
void augmentrotate_delta_gpu(int w, int h, float ALPHA, float *src, float *dest, int ang);

void augment_forward_gpu(int w, int h, float *src, float *dest, float angle, int flip, float scale);
void augment_backward_gpu(int w, int h, float ALPHA, float *src, float *dest, float angle, int flip, float scale);

void augment_forward_max_gpu(int w, int h, int c, int out_w, int out_h, int gap,
                                        float *src, float *dest, int* indexes,
                                        int n_aug,
                                        float* angles, int* flips, float* scales);
void augment_backward_max_gpu(int w, int h, int c, int out_w, int out_h, int gap,
                                        float *src, float *dest, int* indexes);

void augment_forward_split_gpu(int w, int h, int c, int out_w, int out_h, int gap,
                                        float *src, float *dest, int* indexes,
                                        int n_aug,
                                        float* angles, int* flips, float* scales);
void augment_backward_split_gpu(int out_w, int out_h, int out_c,
                                float *src, float *dest, int* indexes);

#endif



#endif
