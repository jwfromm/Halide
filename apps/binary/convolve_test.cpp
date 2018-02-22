// Halide tutorial lesson 10: AOT compilation part 2
// Before reading this file, see lesson_10_aot_compilation_generate.cpp
// This is the code that actually uses the Halide pipeline we've
// compiled. It does not depend on libHalide, so we won't be including
// Halide.h.
//
// Instead, it depends on the header file that lesson_10_generate
// produced when we ran it:
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include "cblas.h"
//#include "BinConvolve.h"
#include "BinConvolveBO.h"
#include "BinConvolveBIBO.h"
#include "BinConvolveBI.h"
#include "BinGemm.h"
#include "BinGemm1A1B.h"
#include "BinGemm2A1B.h"
#include "BinConv1A1W.h"
#include "TestConv.h"
#include "BinDepth.h"
//#include "halide_xnor_train.h"
#include <cmath>
//#include "halide_blas.h"
//#include <valgrind/callgrind.h>
//#include "halide_sgemm.h"

//using namespace Halide;
using namespace std;
using ns = chrono::nanoseconds;
using get_time = chrono::steady_clock;

float pad_mask_check_pixel(int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return false;
    return true;
}

void get_pad_mask(int channels,  int height,  int width,
     int ksize,  int stride, int pad, int64_t* pad_mask)
{
    int c,h,w;
    int height_col = (height - ksize) / stride + 1;
    int width_col = (width - ksize) / stride + 1;
    int filter_size = ksize*ksize*channels;
    int bit;
    int block;
    // pad just indicates that you want your windows to fit in nicely, add however many 0s as is needed (ksize/2) to make that happen,
    // means pad should either be 1 or 0 in cfg file
    if (pad){
        height_col = 1 + (height-1) / stride;
        width_col = 1 + (width-1) / stride;
        pad = ksize/2;
    }
    int output_size = height_col * width_col;
    for (c = 0; c < output_size; ++c) {
        int block_start = c * ((filter_size - 1)/64 + 1);
        int w_offset = (c*stride) % width_col;
        int h_offset = ((c*stride) / width_col) % height_col;
        for (h = 0; h < channels; ++h) {
            for (w = 0; w < (ksize*ksize); ++w) {
                int im_row = h_offset + (w / ksize);
                int im_col = w_offset + (w % ksize);
                int col_offset = (h * ksize*ksize) + w;
                // note that data col is an array of uint64 values, find which uint64 has the bit we want to set
                block = block_start + (col_offset/64);
                // now find the bit in that block that needs to be set
                bit = col_offset % 64;
                // finally, set or clear that bit
                if (pad_mask_check_pixel(height, width, channels, im_row, im_col, h, pad)) {
                    pad_mask[block] |= ((uint64_t) 1 << bit);
                } else {
                    pad_mask[block] &= ~((uint64_t) 1 << bit);
                }
            }
        }
    }
}

void binarize_weights(float *weights, int n, int size, float *binary)
{
    int i, f;
    for(f = 0; f < n; ++f){
        float mean = 0;
        for(i = 0; i < size; ++i){
            mean += fabs(weights[f*size + i]);
        }
        mean = mean / size;
        for(i = 0; i < size; ++i){
            binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
        }
    }   
}

void multibit_cpu(float *input, int size, int bits, float *out) {
    float *carry_over = (float *) malloc(size*sizeof(float));
    memcpy(carry_over, input, size*sizeof(float));
    for (int b = 0; b < bits; b++) {
        float bit_sum = 0;
        for (int i = 0; i < size; i++) {
            bit_sum += fabs(carry_over[i]);
        }
        float bit_mean = bit_sum / size;
        // now that mean is computed update approximation
        for (int i = 0; i < size; i++) {
            if (carry_over[i] > 0) {
                out[i] += bit_mean;
                carry_over[i] = carry_over[i] - bit_mean;
            } else {
                out[i] -= bit_mean;
                carry_over[i] = carry_over[i] + bit_mean;
            }
        }
    }
    free(carry_over);
}

void binarize_cpu(float *input, int n, float *binary)
{
    int i;
    for(i = 0; i < n; ++i){
        binary[i] = (input[i] > 0) ? 1 : -1; 
    }   
}

void fill_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] = ALPHA;
}

void fill_cpu64(int N, float ALPHA, int64_t *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] = ALPHA;
}

void gemm_nn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[i*lda+k];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_nt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            C[i*ldc+j] += sum;
        }
    }
}

void gemm_tn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[k*lda+i];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_tt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
            }
            C[i*ldc+j] += sum;
        }
    }
}


void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    //printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
    int i, j;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
    }
    if(!TA && !TB)
        gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(TA && !TB)
        gemm_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(!TA && TB)
        gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else
        gemm_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
}

void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    gemm_cpu( TA,  TB,  M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc);
}

void scale_bit_sum(float *sums, int num_filters, int output_size, float *alpha, float *beta, int group) {
    int i, j;
    // iterate through each filter
    for (i = group; i < num_filters; i++) {
        // get the alpha value for this filter
        float alpha_val = alpha[i];
        // iterate through the filter values
        for (j = 0; j < output_size; j++) {
            // get a beta value
            float beta_val = beta[j];
            // scale the corresponding output value
            sums[output_size*i + j] *= alpha_val * beta_val;
        }
    }
}

void xnor_and_sum(uint64_t* data_col, uint64_t* filter, int filter_size, int num_filters, int num_outputs, float *output, int group) {
    int i, j;
    int F;
    float sum;
    uint64_t xnor_product;
    for (j = group; j < num_filters; ++j) {
        for (i = 0; i < num_outputs; ++i) {
            F = filter_size;
            sum = 0;
            int filter_block = j * ((filter_size - 1)/64 + 1);
            int window_block = i * ((filter_size - 1)/64 + 1);
            while (F > 64) {
                xnor_product = ~(filter[filter_block++] ^ data_col[window_block++]);
                //sum += bit_sum(xnor_product);
                sum += __builtin_popcountll((long long) xnor_product);
                F -= 64;
            }
            if (F > 0) {
               xnor_product = ~(filter[filter_block] ^ data_col[window_block]) << (64 - F);
               //sum += bit_sum(xnor_product);
               sum += __builtin_popcountll((long long) xnor_product);
            }
            sum = sum - (filter_size - sum);
            output[j*num_outputs + i] = sum;
        }
    }
}

float im2row_get_pixel(uint64_t *im, int height, int width, int channels,
                        int row, int col, int channel, int pad, int bit_offset)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    // compute the block and bit number of the bit we are interested in
    int index = col + width*(row+height*channel) + bit_offset;
    int block = index/64;
    int bit = index % 64;
    return (im[block] & ((uint64_t) 1 << bit));
}

// updated so that data outputs starting at new 64 bit align, after filling out filter_size bits, remainder of current 64bit chunk is garbage
void im2row_cpu_binary(uint64_t* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, uint64_t* data_col, int bit_offset)
{
    int c,h,w;
    int height_col = (height - ksize) / stride + 1;
    int width_col = (width - ksize) / stride + 1;
    int filter_size = ksize*ksize*channels;
    int bit;
    int block;
    // pad just indicates that you want your windows to fit in nicely, add however many 0s as is needed (ksize/2) to make that happen,
    // means pad should either be 1 or 0 in cfg file
    if (pad){
        height_col = 1 + (height-1) / stride;
        width_col = 1 + (width-1) / stride;
        pad = ksize/2;
    }
    int output_size = height_col * width_col;
    for (c = 0; c < output_size; ++c) {
        int block_start = c * ((filter_size - 1)/64 + 1);
        int w_offset = (c*stride) % width_col;
        int h_offset = ((c*stride) / width_col) % height_col;
        for (h = 0; h < channels; ++h) {
            for (w = 0; w < (ksize*ksize); ++w) {
                int im_row = h_offset + (w / ksize);
                int im_col = w_offset + (w % ksize);
                int col_offset = (h * ksize*ksize) + w;
                // note that data col is an array of uint64 values, find which uint64 has the bit we want to set
                block = block_start + (col_offset/64);
                // now find the bit in that block that needs to be set
                bit = col_offset % 64;
                // finally, set or clear that bit
                if (im2row_get_pixel(data_im, height, width, channels, im_row, im_col, h, pad, bit_offset)) {
                    data_col[block] |= ((uint64_t) 1 << bit);
                } else {
                    data_col[block] &= ~((uint64_t) 1 << bit);
                }
            }
        }
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

void binarize_array(float *input, int size, int64_t *binary)
{
    for (int i = 0; i < size; ++i) {
        int index = i;
        int block = index/64;
        int bit = index%64;
        float input_val = input[index];
        if (input_val > 0) {
            binary[block] |= ((uint64_t) 1 << bit);
        } else {
            binary[block] &= ~((uint64_t) 1 << bit);
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

void binarize_filters(float * filters, uint64_t * binary_filters, float *bin_alpha, int n, int c, int size)
{
    int i,f;
    int fsize = c*size*size;
    for (f = 0; f < n; ++f) {
        int block_start = f * ((fsize - 1)/64 + 1);
        float mean = 0;
        for (i = 0; i < fsize; ++i) {
            int index = f*fsize + i;
            int block = block_start + (i / 64);
            int bit = i % 64;
            mean += fabs(filters[index]);
            if (filters[index] > 0) {
                binary_filters[block] |= ((uint64_t) 1 << bit);
            } else {
                binary_filters[block] &= ~((uint64_t) 1 << bit);
            }

        }
        mean = mean / fsize;
        bin_alpha[f] = mean;
    }
}

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


int convolutional_out_height(int h, int size, int stride, bool pad)
{
    if (!pad) h -= size;
    else h -= 1;
    return h/stride + 1;
}

int convolutional_out_width(int w, int size, int stride, bool pad)
{
    if (!pad) w -= size;
    else w -= 1;
    return w/stride + 1;
}

void binarize_A(float *A, float *A_out, float *alpha, int size) {
    int i, j;
    for (i = 0; i < size; i += 64) {
        for (j = 0; j < 64; ++j) {
           alpha[i/64] += abs(A[i + j]); 
        }
        alpha[i/64] = alpha[i/64]/64;
    }
    for (i = 0; i < size; i += 64) {
        for (j = 0; j < 64; ++j) {
            A_out[i + j] = (A[i + j] > 0) ? alpha[i/64] : -alpha[i/64];
        }
    }
}

void binarize_B(float *B, float *B_out, float *beta, int width, int height) {
    int i, j, k;
    int wx = (width - 1)/64 + 1;
    for (i = 0; i < height; ++i) {
        for (j = 0; j < width; j += 64) {
           for (k = 0; k < 64; ++k) {
               beta[wx*i + j/64] += abs(B[i*width + j + k]);
           }
           beta[wx*i + j/64] = beta[wx*i + j/64]/64;
        }
    }
    for (i = 0; i < height; ++i) {
        for (j = 0; j < width; j += 64) {
           for (k = 0; k < 64; ++k) {
               B_out[i*width + j + k] = (B[i*width + j + k] > 0) ? beta[wx*i + j/64] : -beta[wx*i + j/64];
           }
        }
    }
}

void binarize_filter_float(float *A, float *A_bin, float *alpha, int size, int channels, int num_filters) {
    int i, j, f;
    int filter_size = size*size*channels;
    int filter_blocks = (filter_size -1)/64 + 1;
    for (f = 0; f < num_filters; ++f) {
        for (i = 0; i < filter_size; i += 64) {
            for (j = 0; j < 64; ++j) {
                alpha[f*filter_blocks + i/64] += abs(A[f*filter_size + i + j]);
            }
            alpha[f*filter_blocks + i/64] = alpha[f*filter_blocks + i/64]/64;
        }
    }
    for (f = 0; f < num_filters; ++f) {
        for (i = 0; i < filter_size; i += 64) {
            for (j = 0; j < 64; ++j) {
                A_bin[f*filter_size + i + j] = (A[f*filter_size + i + j] > 0) ? alpha[i/64] : -alpha[i/64];
            }
        }
    }
}

void binarize_filter(float *A, int64_t *A_bin, float *alpha, int size, int channels, int num_filters) {
    int i, j, f;
    int filter_size = size*size*channels;
    for (i = 0; i < num_filters; ++i) {
        for (j = 0; j < filter_size; ++j) {
            alpha[i] += abs(A[i*filter_size + j]);
        }
        alpha[i] = alpha[i]/filter_size;
    }
    for (i = 0; i < num_filters*filter_size; i += 64) {
        for (j = 0; j < 64; ++j) {
            if (A[i + j] > 0) {
                A_bin[i/64] |= ((int64_t)1) << j;
            } else {
                A_bin[i/64] &= ~(((int64_t)1) << j);
            }
        }
    }
} 

void xnor_convolve_test(float *input, float *filter, float *alpha, int size, int pad, int stride, int out_x, int out_y, int im_h, int im_w, int channels, int num_filters, float *output) {
    float *binarized_inputs = (float *)calloc(im_h * im_w * channels, sizeof(float));
    int i, j, k, x;
    int filter_blocks = (size * size * channels -1)/64 + 1;
    for (i = 0; i < im_h*im_w*channels; i+=64) {
        for (j = 0; j < 64; ++j) {
            binarized_inputs[i + j] = (input[i + j] > 0) ? 1 : -1;
        }
    }
    float *binarized_filters = (float *)calloc(size * size * channels * num_filters, sizeof(float));
    for (i = 0; i < size*size*channels*num_filters; i+=64) {
        for (j = 0; j < 64; ++j) {
            binarized_filters[i + j] = (filter[i + j] > 0) ? 1 : -1;
            alpha[i/64] += abs(filter[i + j]);
        }
        alpha[i/64] = alpha[i/64]/64;
    }
    float *beta = (float *)calloc(filter_blocks*out_x*out_y, sizeof(float));
    float *im2row_input = (float *)calloc(out_x * out_y * size*size*channels, sizeof(float));
    for (j = 0; j < out_x * out_y; ++j) {
        int w_offset = (j * stride) % out_x;
        int h_offset = ((j * stride) / out_x) % out_y;
        for(i = 0; i < size*size*channels; i+=64) {
            for (x = 0; x < 64; ++x) {
                int im_col = w_offset + ((i+x) % size);
                int im_row = h_offset + ((i+x) / size) % size;
                int im_chan = (i+x) / size / size;
                im2row_input[(i+x) + (j * size*size*channels)] = (input[im_col + im_row*im_w + im_chan*im_w*im_h] > 0) ? 1 : -1;
                beta[i/64 + j * filter_blocks] += abs(input[im_col + im_row*im_w + im_chan*im_w*im_h]);
            }
            beta[i/64 + j*filter_blocks] = beta[i/64 + j*filter_blocks]/64;
        }
    }
    for (k = 0; k < num_filters; ++k) {
        for (j = 0; j < out_x*out_y; ++j) {
            for (i = 0; i < size*size*channels; i += 64) {
                for (x = 0; x < 64; ++x) {
                    output[k*(out_x*out_y) + j] += alpha[k*filter_blocks + i/64] * beta[j*filter_blocks + i/64] * im2row_input[i + x + j*(size*size*channels)] * binarized_filters[i + x + k*(size*size*channels)];             
                }
            }
        }
    }
} 
    
void xnor_convolve_testx(float *input, float *filter, float *alpha, int size, int pad, int stride, int out_x, int out_y, int im_h, int im_w, int channels, int num_filters, float *output) {
    int i, j, k, x;
    int filter_blocks = (size*size*channels-1)/64 + 1;
    uint64_t *binarized_filters = (uint64_t *)calloc(((size * size * channels - 1)/64 + 1) * num_filters, sizeof(uint64_t));
    for (i = 0; i < size*size*channels*num_filters; i+=64) {
        for (j = 0; j < 64; ++j) {
            if (filter[i + j] > 0) {
                binarized_filters[i/64] |= ((uint64_t)1) << j;
            } else {
                binarized_filters[i/64] &= ~(((uint64_t)1) << j);
            }
            alpha[i/64] += abs(filter[i + j]);
        }
        alpha[i/64] = alpha[i/64]/64;
    }
    //float *beta = (float *)calloc(filter_blocks*out_x*out_y, sizeof(float));
    uint64_t *im2row_input = (uint64_t *)calloc(out_x * out_y * filter_blocks, sizeof(uint64_t));
    for (j = 0; j < out_x * out_y; ++j) {
        int w_offset = (j * stride) % out_x;
        int h_offset = ((j * stride) / out_x) % out_y;
        for(i = 0; i < size*size*channels; i+=64) {
            for (x = 0; x < 64; ++x) {
                int im_col = w_offset + ((i+x) % size);
                int im_row = h_offset + ((i+x) / size) % size;
                int im_chan = (i+x) / size / size;
                if (input[im_col + im_row*im_w + im_chan*im_w*im_h] > 0) {
                    im2row_input[i/64 + j*filter_blocks] |= ((uint64_t)1) << x;
                } else {
                    im2row_input[i/64 + j*filter_blocks] &= ~(((uint64_t)1) << x);
                }
                //beta[i/64 + j * filter_blocks] += abs(input[im_col + im_row*im_w + im_chan*im_w*im_h]);
            }
            //beta[i/64 + j*filter_blocks] = beta[i/64 + j*filter_blocks]/64;
        }
    }
    for (k = 0; k < num_filters; ++k) {
        for (j = 0; j < out_x*out_y; ++j) {
            for (i = 0; i < (size*size*channels-1)/64 + 1; ++i) {
                //output[k*(out_x*out_y) + j] += alpha[k*filter_blocks + i] * beta[j*filter_blocks + i] * (2*__builtin_popcountll((long long)(~(im2row_input[j*filter_blocks + i] ^ binarized_filters[k*filter_blocks + i]))) - 64);
                output[k*(out_x*out_y) + j] += alpha[k*filter_blocks + i] * (2*__builtin_popcountll((long long)(~(im2row_input[j*filter_blocks + i] ^ binarized_filters[k*filter_blocks + i]))) - 64);
            }
        }
    }
}

void binarize_data(float *data, int num_points, int64_t *output) {
    int i, j;
    for (i = 0; i < num_points; i+=64) {
        output[i/64] = 0;
        for (j = 0; j < 64; ++j) {
            if (64*i + j < num_points) {
                output[i/64] |= (data[64*i + j] > 0 ? ((int64_t)1) : ((int64_t)0)) << j;
            }
        }
    }
}

void emulate_bgemm(int M, int N, int K, int alpha, int64_t *A, int lda, int64_t *B, int ldb, int beta, float *C, int ldc) {
    int i, j, k;
    for(i = 0; i < M; ++i) {
        for(k = 0; k < K; ++k) {
            register int64_t A_PART = A[i*lda+k];
            for (j = 0; j < N; ++j) {
                C[i*ldc + j] += ((float)__builtin_popcountll(~(A_PART ^ B[k*ldb+j])));
            }
        }
    }
}
        

int main(int argc, char **argv) {
    // Have a look in the header file above (it won't exist until you've run
    // lesson_10_generate).
    // It starts with a definition of a buffer_t:
    //
    // typedef struct buffer_t {
    //     uint64_t dev;
    //     uint8_t* host;
    //     int32_t extent[4];
    //     int32_t stride[4];
    //     int32_t min[4];
    //     int32_t elem_size;
    //     bool host_dirty;
    //     bool dev_dirty;
    // } buffer_t;
    //
    // This is how Halide represents input and output images in
    // pre-compiled pipelines. There's a 'host' pointer that points to the
    // start of the image data, some fields that describe how to access
    // pixels, and some fields related to using the GPU that we'll ignore
    // for now (dev, host_dirty, dev_dirty).
    // Let's make some input data to test with:
    srand (time(NULL));
    int connected_n = 32;
    int connected_k = 32;
    int size=1;
    int stride=1;
    int pad=0;
    //int im_h=75;
    int im_h=4;
    int im_w=4;
    int group = 0;
    int channels= 64;
    int num_filters= 64;
    int batches = 1;
    int out_h = convolutional_out_height(im_h, size, stride, pad);
    int out_w = convolutional_out_width(im_w, size, stride, pad);
    int filter_size = size*size*channels;
    int blocks_per_filter = ((size*size*channels -1)/64) + 1;
    int blocks_per_channel = ((im_h*im_w -1)/64) + 1;
    int connected_kp = (connected_k - 1)/64 + 1;
    float *input = (float *)calloc(batches* channels * im_w * im_h, sizeof(float));
    float *input2 = (float *)calloc(batches* channels * im_w * im_h, sizeof(float));
    for (int b = 0; b < batches; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int y = 0; y < im_h; y++) {
                for (int x = 0; x < im_w; x++) {
                    input[b*channels*im_h*im_w + c*im_h*im_w + y * im_w + x] = ((float)rand())/(RAND_MAX)-.5f;
                }
            }
        }
    }
    float *filter = (float *) calloc(size * size * channels * num_filters, sizeof(float));
    float *filter2 = (float *) calloc(size * size * channels * num_filters, sizeof(float));
    int64_t *bin_filter = (int64_t *) calloc(((size * size * channels - 1)/64 + 1) * num_filters, sizeof(int64_t));
    for (int f = 0; f < num_filters; ++f) {
        for (int c = 0; c < channels; ++c) {
            for (int y = 0; y < size; ++y) {
                for (int x = 0; x < size; ++x) {
                    filter[f*channels*size*size + c*size*size + y*size + x] = 2*(((float)rand())/(RAND_MAX) - .5f);
                }
            }
        }
    }

    float *connected_A = (float *) calloc(connected_k, sizeof(float));
    float *bin_connected_A = (float *) calloc(connected_k, sizeof(float));
    int64_t *bin_B = (int64_t *) calloc(connected_k*connected_n/64, sizeof(int64_t));
    float *connected_B = (float *) calloc(connected_k*connected_n, sizeof(float));
    float *bin_connected_B = (float *) calloc(connected_k*connected_n, sizeof(float));

    for (int i = 0; i < connected_k; ++i) {
        connected_A[i] = 2*(((float)rand())/(RAND_MAX) - .5f);
        for (int j = 0; j < connected_n; ++j) {
            connected_B[i*connected_n + j] = 2*(((float)rand())/(RAND_MAX) - .5f);
        }
    }

    // And the memory where we want to write our output:
    int output_size = min(out_w * out_h * num_filters * batches, connected_n);
    float *output = (float *) calloc(output_size, sizeof(float)); 
    float *output_test = (float *) calloc(output_size, sizeof(float));
    for (int b = 0; b < batches; ++b) {
        for (int c = 0; c < num_filters; ++c) {
            for (int y = 0; y < out_h; ++y) {
                for (int x = 0; x < out_w; ++x) {
                    output[b*num_filters*out_h*out_w + c*out_h*out_w + y*out_w + x] = 0;
                    output_test[c*out_h*out_w + y*out_w + x] = 0;
                }
            }
        }
    }

    int binary_input_depth = (size*size*channels-1)/64 + 1;
    int binary_input_size = binary_input_depth * out_w * out_h;

    //float *workspace = (float *)calloc(out_w*out_h*size*size*channels, sizeof(float));
    float *workspace2 = (float *)calloc(out_w*out_h*size*size*channels, sizeof(float));
    //int64_t *bin_workspace = (int64_t *)calloc(out_w*out_h*size*size*channels, sizeof(int64_t));
    //int64_t *bin_workspace2 = (int64_t *)calloc(out_w*out_h*size*size*channels, sizeof(int64_t));
    int64_t *pad_mask = (int64_t *)calloc(out_w*out_h*size*size*channels/64, sizeof(int64_t));
    for (int i = 0; i < out_w*out_h*size*size*channels; ++i) {
        workspace2[i] = 0;
    }

// prepare to test conv layers
    int filter_compressed = (filter_size - 1)/64 + 1;
    float *alpha_conv = (float *)calloc(num_filters * filter_compressed, sizeof(float));
    fill_cpu(num_filters * filter_compressed, 0, alpha_conv, 1);
    float *alpha_test = (float *)calloc(num_filters * filter_compressed, sizeof(float));
    fill_cpu(num_filters * filter_compressed, 0, alpha_test, 1);

    int64_t *bin_filters = (int64_t *)calloc(num_filters * filter_compressed, sizeof(int64_t));
    fill_cpu64(num_filters * filter_compressed, 0, bin_filters, 1);

    
    int n = out_h*out_w;
    int m = num_filters;
    int k = size*size*channels;
    cout<<n<<","<<m<<","<<k; 
    binarize_cpu(filter, num_filters*size*size*channels, filter2);
    binarize_cpu(input, channels*im_h*im_w*batches, input2);
    //auto start = get_time::now();
    //im2col_cpu(input2, channels, im_h, im_w, size, stride, pad, workspace);
    //auto end = get_time::now();
    //auto diff = end - start;
    //auto prepR = diff;
    //cout<<"Elapsed time for im2col is : "<<chrono::duration_cast<ns>(diff).count()<<" ns "<<endl;
    //sgemm(1, &halide_workspace_buf, &halide_filter_buf, 1, &halide_output_buf, &halide_output_buf);
        //binarize_array(filter, bin_filters, num_filters*size*size*channels);
    //binarize_array(workspace2, bin_workspace, out_w*out_h*size*size*channels);
    //CALLGRIND_START_INSTRUMENTATION;

    auto start = get_time::now(); 

    // only do this if testing gemm
    //im2col_cpu(input, channels, im_h, im_w, size, stride, pad, workspace);
    //binarize_cpu(workspace, out_h*out_w*size*size*channels, workspace2);
    //cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, m, k, 1, workspace2, n, filter2, k, 1, output, n); 

    //otherwise do this
    im2col_cpu(input2, channels, im_h, im_w, size, stride, pad, workspace2);
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, m, k, 1, workspace2, n, filter2, k, 1, output, n); 

    auto end = get_time::now();
    auto diff = end - start;
    //im2col_cpu(input, channels, im_h, im_w, size, stride, pad, workspace);
    cout<<"Elapsed time for cblas_sgemm is : "<<chrono::duration_cast<ns>(diff).count()/1000000<<" ms "<<endl;
    //auto compR = diff;
    //fill_cpu(out_w * out_h * num_filters, 0, output, 1);

    //BinConvolve tester = BinConvolve(im_w, im_h, channels, num_filters, size, stride, pad);
    //BinConvolveBO tester = BinConvolveBO(im_w, im_h, channels, num_filters, size, stride, pad);
    //BinConvolveBIBO tester2 = BinConvolveBIBO(out_w, out_h, num_filters, num_filters, size, stride, pad);
    //tester.realize(input, filter, bin_workspace);
    //tester2.realize(bin_workspace, filter, bin_workspace2);
    //tester.realize(input, filter, output_test);
    //BinGemm gemmtester = BinGemm(m, n, k);
    //gemmtester.realize(filter, workspace, output_test);

    // test of cntk bin convolve stuff
    binarize_array(filter, num_filters*size*size*channels, bin_filters);
    //binarize_filters_depth(filter, size, channels, num_filters, bin_filters);
    //get_pad_mask(channels, im_h, im_w, size, stride, pad, pad_mask);
    //BinConv1A1W tester = BinConv1A1W(bin_filters, pad_mask, im_w, im_h, channels, num_filters, size, stride, pad);
    //TestConv tester = TestConv(filter2, im_w, im_h, channels, num_filters, size, stride, pad);
    //BinDepth tester = BinDepth(bin_filters, im_w, im_h, channels, num_filters, size, stride, pad);
    BinGemm1A1B tester = BinGemm1A1B(bin_filters, im_w*im_h, num_filters, channels, true);
    tester.realize(input, output_test);
    tester.realize(input, output_test);
    start = get_time::now(); 
    //invoke_halide_convolve(filter, input, num_filters, size, channels, pad, stride, im_w, im_h, output_test);
    //tester.realize(input, filter, output_test);
    tester.realize(input, output_test);
    //gemmtester.realize(filter, workspace, output_test);
    end = get_time::now();
    diff = end - start;
    cout<<"Elapsed time for halide_binconvolve is : "<<chrono::duration_cast<ns>(diff).count()/1000000<<" ms "<<endl;
    //auto compB = diff;
    //cout<<"Speed ratio is : "<<(compR/compB)<<endl;
    //cout<<"Prep ratio is : "<<(prepR/prepB)<<endl;
    //cout<<"Cumulative ratio is : "<<((prepR+compR)/(prepB+compB))<<endl;
    //CALLGRIND_STOP_INSTRUMENTATION; 
    //CALLGRIND_DUMP_STATS;

    // correctness check for approximated convolutional layers
    //emulate_bgemm(m, n, (k-1)/64 + 1, 1, bin_filter, (k-1)/64 + 1, bin_workspace, n, 1, output, n);
    for (int c = 0; c < num_filters; ++c) {
        for (int y = 0; y < out_h; y++) {
            for (int x = 0; x < out_w; x++) {
                float output_val = output[c*out_h*out_w + y * out_w + x];
                float correct_val = output_test[c*out_h*out_w + y * out_w + x];
                if (abs(output_val - correct_val) > 0.001f) {
                   printf("output(%d, %d, %d) regular val is %f halide val is %f\n", x, y, c, output_val, correct_val);
                }
            }
        }
    }

    binarize_array_T(connected_B, connected_n, connected_k, bin_B);
    BinGemm1A1B multitester = BinGemm1A1B(bin_B, 1, connected_n, connected_k);
    multitester.realize(connected_A, output_test);
    multitester.realize(connected_A, output_test);
    start = get_time::now(); 
    // test bingemm2a1b
    multitester.realize(connected_A, output_test);
    end = get_time::now();
    diff = end - start;
    cout<<"Elapsed time for halide bitgemm1a1b is : "<<chrono::duration_cast<ns>(diff).count()/1000<<" us "<<endl;

    binarize_cpu(connected_B, connected_n*connected_k, bin_connected_B);
    binarize_cpu(connected_A, connected_k, bin_connected_A);
    //multibit_cpu(connected_A, connected_k, 2, bin_connected_A);

    memset(output, 0, connected_n*1*sizeof(float));
    start = get_time::now(); 
    //gemm(0, 0, 1, connected_n, connected_k, 1, connected_A, connected_k, connected_B, connected_n, 1, output, connected_n);
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, connected_n, 1, connected_k, 1, bin_connected_B, connected_n, bin_connected_A, connected_k, 1, output, connected_n); 
    end = get_time::now();
    diff = end - start;
    cout<<"Elapsed time for cblas gemm is : "<<chrono::duration_cast<ns>(diff).count()/1000<<" us "<<endl;

    for (int x = 0; x < connected_n; x++) {
        float output_val = output[x];
        float correct_val = output_test[x];
        if (abs(output_val - correct_val) > 0.001f) {
           //printf("GEMM failed: output(%d) regular val is %f halide val is %f\n", x, output_val, correct_val);
        }
    }
    
    //multibit testing
    //float test_array[] = {1,2,3,4,5,6,7,8,9,10};
    //float out_array[10] = {};
    //MultiTest multitester = MultiTest(10, 1);
    //multitester.realize(test_array, out_array);

    // speed test for halide convolutional layer
    /* 
    start = get_time::now();
    im2col_cpu(input, channels, im_h, im_w, size, stride, pad, workspace);
    hblas_sgemm(HblasColMajor, HblasNoTrans, HblasNoTrans, n, m, k, 1, workspace, n, filter, k, 1, output_test, n); 
    end = get_time::now();
    diff = end - start;
    cout<<"Elapsed time for hblas convolve is : "<<chrono::duration_cast<ns>(diff).count()<<" ns "<<endl;

    start = get_time::now();
    //halide_xnor_fast(&input_buf, &filter_buf, group, filter_size, size, pad, stride, out_w, out_h, &output_buf);
    //hblas_xnor(input, filter, im_w, im_h, num_filters, channels, filter_size, size, pad, stride, out_w, out_h, output);
    end = get_time::now();
    diff = end - start;
    cout<<"Elapsed time for simple halide convolve is : "<<chrono::duration_cast<ns>(diff).count()<<" ns "<<endl;
    */

 // Everything worked!
    printf("Success!\n");
    return 0;
}
