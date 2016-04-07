#ifndef sidebyside_LAYER_H
#define sidebyside_LAYER_H

#include "layer.h"
#include "network.h"

layer make_sidebyside_layer(int batch, int splits, int gap, int w, int h, int c);
void forward_sidebyside_layer(const layer l, network_state state);
void backward_sidebyside_layer(const layer l, network_state state);

#ifdef GPU
void forward_sidebyside_layer_gpu(const layer l, network_state state);
void backward_sidebyside_layer_gpu(const layer l, network_state state);
#endif

#endif
