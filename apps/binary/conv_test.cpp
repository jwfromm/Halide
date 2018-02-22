#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "cblas.h"
#include "BinGemm1A1B.h"
#include "BinDepth.h"

using namespace std;
using ns = chrono::nanoseconds;
using get_time = chrono::steady_clock;

float im2col_get_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu(float* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, float* data_col)
{
    int c,h,w;
    int height_col = (height - ksize) / stride + 1;
    int width_col = (width - ksize) / stride + 1;
    if (pad){
        height_col = 1 + (height-1) / stride;
        width_col = 1 + (width-1) / stride;
        pad = ksize/2;
    }
    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                float pixel_val = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad); 
                data_col[col_index] = pixel_val;
            }
        }
    }
}

int convolutional_out_size(int x, int size, int stride, bool pad)
{
    if (!pad) x -= size;
    else x -= 1;
    return x/stride + 1;
}

void binarize_cpu(float *input, int n, float *binary)
{
    int i;
    for(i = 0; i < n; ++i){
        binary[i] = (input[i] > 0) ? 1 : -1; 
    }   
}

void binarize_filters_depth(float *w, int size, int channels, int num_filters, int64_t *binary)
{
    int index;
    int bin_c_block;
    int bin_c_bit;
    int bin_block_index;
    float input_val;
    for (int k = 0; k < num_filters; ++k) {
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                for (int c = 0; c < channels; ++c) {
                    index = k*size*size*channels + c*size*size + j*size + i;
                    bin_c_block = c/64;
                    bin_c_bit = c%64;
                    bin_block_index = k*size*size*(channels/64) + bin_c_block*size*size + j*size + i;
                    input_val = w[index];
                    if (input_val > 0) {
                        binary[bin_block_index] |= ((uint64_t) 1 << bin_c_bit);
                    } else {
                        binary[bin_block_index] &= ~((uint64_t) 1 << bin_c_bit);
                    }
                }
            }
        }
    }
}

void binarize_array_T(float *input, int x, int y, int64_t *binary)
{
    for (int i = 0; i < x; ++i) {
        for (int j = 0; j < y; ++j) {
            int index = y*i + j;
            int block = index/64;
            int bit = index%64;
            float input_val = input[j*x + i];
            if (input_val > 0) {
                binary[block] |= ((uint64_t) 1 << bit);
            } else {
                binary[block] &= ~((uint64_t) 1 << bit);
            }
        }
    }
}

void rand_init(float *input, int size) {
    for (int i = 0; i < size; ++i) {
       input[i] = 2*(((float)rand())/(RAND_MAX) - .5f);
    }
} 

int main(int argc, char **argv) {
    srand (time(NULL));
    int w = 32;
    int h = 32;
    int channels = 512;
    int num_filters = 512;
    int size = 3;
    int stride = 1;
    int pad = 0;
    int out_h = convolutional_out_size(h, size, stride, pad);
    int out_w = convolutional_out_size(w, size, stride, pad);

    // initialize variables
    float *input = (float *) calloc(channels*w*h, sizeof(float));
    rand_init(input, channels*w*h);
    float *input2 = (float *) calloc(channels*w*h, sizeof(float));
    float *filter = (float *) calloc(size*size*channels*num_filters, sizeof(float));
    rand_init(filter, size*size*channels*num_filters);
    float *filter2 = (float *) calloc(size*size*channels*num_filters, sizeof(float));
    int64_t  *bin_filters = (int64_t *) calloc(((size*size*channels)/64)*num_filters, sizeof(int64_t));

    float *workspace = (float *) calloc(out_w*out_h*size*size*channels, sizeof(float));

    float *output = (float *)calloc(out_w*out_h*num_filters, sizeof(float));
    float *output_test = (float *)calloc(out_w*out_h*num_filters, sizeof(float));

    // test halide gemm
    binarize_filters_depth(filter, size, channels, num_filters, bin_filters);
    BinDepth tester = BinDepth(bin_filters, w, h, channels, num_filters, size, stride, pad);
    //tester.realize(input, output);
    //tester.realize(input, output);
    auto start = get_time::now();
    tester.realize(input, output);
    auto end = get_time::now();
    auto diff = end - start;
    cout<<"Halide gemm : "<<chrono::duration_cast<ns>(diff).count()/1000<<" us "<<endl;

    // test cblas gemm
    int n = out_w*out_h;
    int m = num_filters;
    int k = size*size*channels;
    binarize_cpu(filter, num_filters*size*size*channels, filter2);
    binarize_cpu(input, channels*h*w, input2);

    start = get_time::now();
    im2col_cpu(input2, channels, h, w, size, stride, pad, workspace);
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, m, k, 1, workspace, n, filter2, k, 1, output_test, n); 
    end = get_time::now();
    diff = end - start;
    cout<<"BLAS gemm : "<<chrono::duration_cast<ns>(diff).count()/1000<<" us "<<endl;

    for (int c = 0; c < num_filters; ++c) {
        for (int y = 0; y < out_h; y++) {
            for (int x = 0; x < out_w; x++) {
                float output_val = output[c*out_h*out_w + y * out_w + x];
                float correct_val = output_test[c*out_h*out_w + y * out_w + x];
                if (abs(output_val - correct_val) > 0.001f) {
                   printf("output(%d, %d, %d) halide val is %f blas val is %f\n", x, y, c, output_val, correct_val);
                }
            }
        }
    }
}
