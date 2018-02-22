#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "cblas.h"
#include "BinGemm1A1B.h"

using namespace std;
using ns = chrono::nanoseconds;
using get_time = chrono::steady_clock;

void binarize_cpu(float *input, int n, float *binary)
{
    int i;
    for(i = 0; i < n; ++i){
        binary[i] = (input[i] > 0) ? 1 : -1; 
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
    int connected_n = 256;
    int connected_k = 256;
    int connected_m = 32*32; 

    // initialize variables
    float *connected_A = (float *) calloc(connected_m*connected_k, sizeof(float));
    rand_init(connected_A, connected_k);
    float *bin_connected_A = (float *) calloc(connected_m*connected_k, sizeof(float));
    int64_t *bin_B = (int64_t *) calloc(connected_k*connected_n/64, sizeof(int64_t));
    float *connected_B = (float *) calloc(connected_k*connected_n, sizeof(float));
    rand_init(connected_B, connected_k*connected_n);
    float *bin_connected_B = (float *) calloc(connected_k*connected_n, sizeof(float));

    float *output = (float *)calloc(connected_m*connected_n, sizeof(float));
    float *output_test = (float *)calloc(connected_m*connected_n, sizeof(float));

    // test halide gemm
    binarize_array_T(connected_B, connected_n, connected_k, bin_B);
    BinGemm1A1B tester = BinGemm1A1B(bin_B, connected_m, connected_n, connected_k);
    //tester.realize(connected_A, output);
    tester.realize(connected_A, output);
    auto start = get_time::now();
    tester.realize(connected_A, output);
    auto end = get_time::now();
    auto diff = end - start;
    cout<<"Halide gemm : "<<chrono::duration_cast<ns>(diff).count()/1000<<" us "<<endl;

    // test cblas gemm
    binarize_cpu(connected_B, connected_n*connected_k, bin_connected_B);
    binarize_cpu(connected_A, connected_m*connected_k, bin_connected_A);
    start = get_time::now();
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, connected_n, connected_m, connected_k, 1, bin_connected_B, connected_n, bin_connected_A, connected_k, 1, output_test, connected_n); 
    end = get_time::now();
    diff = end - start;
    cout<<"BLAS gemm : "<<chrono::duration_cast<ns>(diff).count()/1000<<" us "<<endl;

    // check results
    for (int x = 0; x < connected_m*connected_n; ++x) {
        float output_val = output[x];
        float test_val = output_test[x];
        if (abs(output_val - test_val) > 0.001f) {
           printf("GEMM failed: output(%d) halide val is %f blas val is %f\n", x, output_val, test_val);
        }
    }
}
