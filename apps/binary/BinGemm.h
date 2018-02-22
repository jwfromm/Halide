#ifndef BINGEMM
#define BINGEMM

#include "Halide.h"

using namespace Halide;

class BinGemm {
    Buffer<float> A;
    Buffer<float> B;
    int n;
    int m;
    int p;
    int bin_width;
    Func output;
    Target t;
public:
    BinGemm(int m, int n, int p, bool gpu = false) :
    A(Buffer<float>(p,m)),
    B(Buffer<float>(n,p)),
    n(n),
    m(m),
    p(p),
    bin_width((p - 1)/64 + 1),
    t(get_host_target())
    {
        Var x("x"), y("y"), k("k");

        Func Bt("transpose B");
        Bt(x, y) = B(y, x);

        Func binarize_A("binarize_A");
        RDom r(0, 64);
        RDom bw(0, bin_width);
        
        binarize_A(x, y) = sum(select(A(64*x + r, y) > 0, cast<int64_t>(1) << r.x, cast<int64_t>(0)), "compress_A"); 

        Func binarize_B("binarize_B");
        binarize_B(x, y) = sum(select(Bt(64*x + r, y) > 0, cast<int64_t>(1) << r.x, cast<int64_t>(0)), "compress_B"); 

        Func xnor("xnor");
        xnor(k, x, y) = popcount(binarize_A(k, y) ^ binarize_B(k, x));

        output(x, y) = -((2 * cast<float>(sum(xnor(bw.x, x, y), "accumulate"))) - (64*bin_width));

        if (!gpu) {
            output.reorder(y, x);
            binarize_A.compute_root();
            binarize_A.vectorize(x, 8);
            output.vectorize(y, 8);
            output.parallel(x, 8);
            binarize_B.compute_at(output, x);
            //t.set_feature(Target::Profile);
        } else {
            Var block, thread;
            binarize_A.compute_root();
            binarize_A.gpu_blocks(y).gpu_threads(x);
            output.reorder(y, x);
            binarize_B.compute_at(output, x);
            output.gpu_blocks(x).gpu_threads(y);
            t.set_feature(Target::CUDA);
        }
        output.compile_jit(t);
        
    }

    void realize(float *A_array, float *B_array, float *out_array) {
        Buffer<float> outbuf = Buffer<float>(out_array, n, m);
        std::memcpy(A.get()->data(), A_array, m*p*sizeof(float));
        std::memcpy(B.get()->data(), B_array, p*n*sizeof(float));
        output.realize(outbuf);
    }
};

#endif
