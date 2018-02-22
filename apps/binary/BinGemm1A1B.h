#ifndef BINGMM1A1B
#define BINGEMM1A1B

#include "Halide.h"

using namespace Halide;

class BinGemm1A1B {
    Buffer<float> A;
    Buffer<int64_t> B;
    int n;
    int m;
    int k;
    int bin_width;
    Func output;
    Target t;
public:
    BinGemm1A1B(int64_t *B_in, int m, int n, int k, bool trans_A = false) :
    A(trans_A ? Buffer<float>(m, k) : Buffer<float>(k,m)),
    B(Buffer<int64_t>(B_in, k/64,n)),
    n(n),
    m(m),
    k(k),
    bin_width((k - 1)/64 + 1),
    t(get_host_target())
    {
        Var x("x"), y("y"), p("p");

        Func binarize_B("binarize_B");
        //Func Bt("transpose B");
        //Bt(x, y) = B(y, x);
        binarize_B(x, y) = B(x, y);

        Func binarize_A("binarize_A");
        RDom r(0, 64);
        RDom bw(0, bin_width);
        //RDom rsum(A);

        Func A_in;
        if (trans_A) {
            A_in(x, y) = A(y, x);
        } else {
            A_in(x, y) = A(x, y);
        }
        
        binarize_A(x, y) = sum(select(A_in(64*x + r, y) > 0, cast<int64_t>(1) << r.x, cast<int64_t>(0)), "compress_A"); 
        // compute first bit mean
        //mean1(x) = cast<float>(0);
        //mean1(0) = sum(abs(At(rsum.y, rsum.x)), "mean1_sum") / (m*p);

        //Func A2_remainder("A2_remainder");
        //A2_remainder(x, y) = At(x, y) - select(At(x, y) > 0, mean1(0), -mean1(0));

        //Func binarize_A2("binarize_A2");
        //Func mean2("mean2");
        //binarize_A2(x, y) = sum(select(A2_remainder(64*x + r, y) > 0, cast<int64_t>(1) << r.x, cast<int64_t>(0)), "compress_A2");
        //mean2(x) = cast<float>(0);
        //mean2(0) = sum(abs(A2_remainder(rsum.y, rsum.x)), "mean2_sum") / (m*p);

        //binarize_B(x, y) = sum(select(Bt(64*x + r, y) > 0, cast<int64_t>(1) << r.x, cast<int64_t>(0)), "compress_B"); 

        Func xnor("xnor");
        xnor(p, x, y) = popcount(binarize_A(p, y) ^ binarize_B(p, x));
        //xnor2(k, x, y) = popcount(binarize_A2(k, y) ^ binarize_B(k, x));

        output(x, y) = (-((2 * cast<float>(sum(xnor(bw.x, x, y), "accumulate"))) - (64*bin_width)));

        /*binarize_A1.compute_root();
        binarize_A1.vectorize(x, 8);
        binarize_A2.compute_root();
        binarize_A2.vectorize(x, 8);
        mean1.compute_root();
        mean2.compute_root();
        binarize_B.compute_at(output, x);
        output.vectorize(x, 8);
        */
        //mean1.compute_root();
        //mean1.vectorize(x, 8);
        //mean2.compute_root();
        //mean2.vectorize(x, 8);
        //binarize_A.compute_at(output, y);
        binarize_A.compute_root();
        output.vectorize(x, 16);
        //output.parallel(y, 8);
        //binarize_A2.compute_at(output, y);
        //Var x_inner, x_outer;
        //output.split(x, x_outer, x_inner, 8);
        //output.vectorize(x_inner);
        //output.parallel(x_outer);
        //t.set_feature(Target::Profile);
        output.compile_jit(t);
        
    }

    void realize(float *A_array, float *out_array) {
        Buffer<float> outbuf = Buffer<float>(out_array, n, m);
        std::memcpy(A.get()->data(), A_array, m*k*sizeof(float));
        output.realize(outbuf);
    }
};

#endif
