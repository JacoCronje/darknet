#ifndef keypoint_LAYER_H
#define keypoint_LAYER_H

#include "layer.h"
#include "network.h"

layer make_keypoint_layer(int batch, int w, int h, int c, char *cfgfile, char* weightfile);
void forward_keypoint_layer(const layer l, network_state state);
void backward_keypoint_layer(const layer l, network_state state);

#ifdef GPU
void forward_keypoint_layer_gpu(const layer l, network_state state);
void backward_keypoint_layer_gpu(const layer l, network_state state);

void keypoint_gpu(int w, int h, float tx, float ty, float *dest);
#endif



#endif
