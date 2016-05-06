#ifndef COMPACT_LAYER_H
#define COMPACT_LAYER_H

#include "layer.h"
#include "network.h"

layer make_compact_layer(int batch, int splits, int method, int w, int h, int c);
void forward_compact_layer(const layer l, network_state state);
void backward_compact_layer(const layer l, network_state state);

#ifdef GPU
void forward_compact_layer_gpu(const layer l, network_state state);
void backward_compact_layer_gpu(const layer l, network_state state);

void compact_forward_max_gpu(int w, int h, int c, int splits, float *src, float *dest, int *indexes);
void compact_backward_max_gpu(int w, int h, int c, int splits, float *src, float *dest, int *indexes);
void compact_forward_padd_gpu(int w, int h, int c, float *src, float *dest);
void compact_backward_padd_gpu(int w, int h, int c, float *src, float *dest);
void compact_forward_pmax_gpu(int w, int h, int c, float *src, float *dest, int *indexes);
void compact_backward_pmax_gpu(int w, int h, int c, float *src, float *dest, int *indexes);
#endif

#endif
