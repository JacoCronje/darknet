#ifndef COMPACT_LAYER_H
#define COMPACT_LAYER_H

#include "layer.h"
#include "network.h"

layer make_compact_layer(int batch, int splits, int w, int h, int c);
void forward_compact_layer(const layer l, network_state state);
void backward_compact_layer(const layer l, network_state state);

#ifdef GPU
void forward_compact_layer_gpu(const layer l, network_state state);
void backward_compact_layer_gpu(const layer l, network_state state);
#endif

#endif
