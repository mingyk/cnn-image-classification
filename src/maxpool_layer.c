#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "dubnet.h"


// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
tensor forward_maxpool_layer(layer *l, tensor x)
{
    // Saving our input
    // Probably don't change this
    tensor_free(l->x);
    l->x = tensor_copy(x);

    assert(x.n == 4);

    tensor y = tensor_vmake(4,
        x.size[0],  // same # data points and # of channels (N and C)
        x.size[1],
        (x.size[2]-1)/l->stride + 1, // H and W scaled based on stride
        (x.size[3]-1)/l->stride + 1);

    // This might be a useful offset...
    int pad = -((int) l->size - 1)/2;

    // TODO: 6.1 - iterate over the input and fill in the output with max values
    int im_n = x.size[0];
    int im_c = x.size[1];
    int im_h = x.size[2];
    int im_w = x.size[3];
    int size = l->size;
    int stride = l->stride;
    int res_h = (x.size[2]-1)/stride + 1;
    int res_w = (x.size[3]-1)/stride + 1;
    
    for(int batch = 0; batch < im_n; ++batch){
        for(int c = 0; c < im_c; ++c){
            int index = 0;
            for(int h = pad; h < im_h + -pad - size + 1; h+=stride){
                for(int w = pad; w < im_w + -pad - size + 1; w+=stride){
                    float max_val = -FLT_MAX;
                    for(int kr = 0; kr < size; ++kr){
                        for(int kc = 0; kc < size; ++kc){
                            int xc = w + kc;
                            int yc = h + kr;
                            float val;
                            if(xc < 0 || yc < 0 || xc >= im_w || yc >= im_h){
                                val = -FLT_MAX;
                            }else{
                                val = x.data[im_w*im_h*im_c*batch + im_w*im_h*c + im_w*yc + xc];
                            }
                            max_val = max_val < val ? val : max_val;
                        }
                    }
                    int idx = res_w*res_h*im_c*batch + res_w*res_h*c + index;
                    y.data[idx] = max_val;
                    index++;
                }
            }
        }
    }

    return y;
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix dy: error term for the previous layer
tensor backward_maxpool_layer(layer *l, tensor dy)
{
    tensor x    = l->x;
    tensor dx = tensor_make(x.n, x.size);
    int pad = -((int) l->size - 1)/2;

    // TODO: 6.2 - find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.
    int im_n = x.size[0];
    int im_c = x.size[1];
    int im_h = x.size[2];
    int im_w = x.size[3];
    int size = l->size;
    int stride = l->stride;
    int res_h = (x.size[2]-1)/stride + 1;
    int res_w = (x.size[3]-1)/stride + 1;
    
    for(int batch = 0; batch < im_n; ++batch){
        for(int c = 0; c < im_c; ++c){
            int index = 0;
            for(int h = pad; h < im_h + -pad - size + 1; h+=stride){
                for(int w = pad; w < im_w + -pad - size + 1; w+=stride){
                    float max_val = -FLT_MAX;
                    long max_index = 0;
                    for(int kr = 0; kr < size; ++kr){
                        for(int kc = 0; kc < size; ++kc){
                            int xc = w + kc;
                            int yc = h + kr;
                            float val;
                            long idx;
                            if(xc < 0 || yc < 0 || xc >= im_w || yc >= im_h){
                                val = -FLT_MAX;
                                idx = 0;
                            }else{
                                idx = im_w*im_h*im_c*batch + im_w*im_h*c + im_w*yc + xc;
                                val = x.data[idx];
                            }
                            if(max_val < val){
                                max_val = val;
                                max_index = idx;
                            }
                        }
                    }
                    dx.data[max_index] += dy.data[res_w*res_h*im_c*batch + res_w*res_h*c + index];
                    index++;
                }
            }
        }
    }

    return dx;
}

// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer *l, float rate, float momentum, float decay){}

// Make a new maxpool layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of maxpool filter to apply
// int stride: stride of operation
layer make_maxpool_layer(size_t size, size_t stride)
{
    layer l = {0};
    l.size = size;
    l.stride = stride;
    l.forward  = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update   = update_maxpool_layer;
    return l;
}

