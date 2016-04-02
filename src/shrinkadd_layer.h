#ifndef shrinkadd_LAYER_H
#define shrinkadd_LAYER_H

#include "layer.h"
#include "network.h"

layer make_shrinkadd_layer(int batch, int splits, int w, int h, int c);
void forward_shrinkadd_layer(const layer l, network_state state);
void backward_shrinkadd_layer(const layer l, network_state state);

#ifdef GPU
void forward_shrinkadd_layer_gpu(const layer l, network_state state);
void backward_shrinkadd_layer_gpu(const layer l, network_state state);
#endif

#endif
