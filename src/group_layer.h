#ifndef group_layer_H
#define group_layer_H
#include "network.h"
#include "layer.h"

typedef layer group_layer;

group_layer make_group_layer(int batch, int n, int *output_channels, int w, int h, int c);
void forward_group_layer(const group_layer l, network_state state);
void backward_group_layer(const group_layer l, network_state state);

#ifdef GPU
void forward_group_layer_gpu(const group_layer l, network net);
void backward_group_layer_gpu(const group_layer l, network net);
#endif

#endif
