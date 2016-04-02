#ifndef shrinkmax_LAYER_H
#define shrinkmax_LAYER_H

#include "layer.h"
#include "network.h"

layer make_shrinkmax_layer(int batch, int splits, int w, int h, int c);
void forward_shrinkmax_layer(const layer l, network_state state);
void backward_shrinkmax_layer(const layer l, network_state state);

#ifdef GPU
void forward_shrinkmax_layer_gpu(const layer l, network_state state);
void backward_shrinkmax_layer_gpu(const layer l, network_state state);
#endif

#endif
