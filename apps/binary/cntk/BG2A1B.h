#ifndef BINGEMM2A1B
#define BINGEMM2A1B

#include "Halide.h"

using namespace Halide;

class BinGemm2A1B {
    Buffer<float> A;
    Buffer<int64_t> B;
    int n;
    int m;
    int p;
    int bin_width;
    Func output;
    Target t;
public:
    BinGemm2A1B(int64_t *B_in, int m, int n, int p, bool gpu = false) :
    A(Buffer<float>(p,m)),
    //B(Buffer<int64_t>(n/64,p)),
    B(Buffer<int64_t>(B_in, p/64, n)),
    n(n),
    m(m),
    p(p),
    bin_width((p - 1)/64 + 1),
    t(get_host_target())
    {
        Var x("x"), y("y"), k("k");
    
        Func At("At");

        Func binarize_B("binarize_B");
        //Func Bt("transpose B");
        //Bt(x, y) = B(y, x);
        binarize_B(x, y) = B(x, y);

        Func binarize_A1("binarize_A1"), mean1("mean1");
        RDom r(0, 64);
        RDom bw(0, bin_width);
        RDom rsum(A);
        
        binarize_A1(x, y) = sum(select(A(64*x + r, y) > 0, cast<int64_t>(1) << r.x, cast<int64_t>(0)), "compress_A1"); 
        // compute first bit mean
        mean1(x) = cast<float>(0);
        mean1(0) = sum(abs(A(rsum.x, rsum.y)), "mean1_sum") / (m*p);

        Func A2_remainder("A2_remainder");
        A2_remainder(x, y) = A(x, y) - select(A(x, y) > 0, mean1(0), -mean1(0));

        Func binarize_A2("binarize_A2");
        Func mean2("mean2");
        binarize_A2(x, y) = sum(select(A2_remainder(64*x + r, y) > 0, cast<int64_t>(1) << r.x, cast<int64_t>(0)), "compress_A2");
        mean2(x) = cast<float>(0);
        mean2(0) = sum(abs(A2_remainder(rsum.x, rsum.y)), "mean2_sum") / (m*p);

        //binarize_B(x, y) = sum(select(Bt(64*x + r, y) > 0, cast<int64_t>(1) << r.x, cast<int64_t>(0)), "compress_B"); 

        Func xnor1("xnor1"), xnor2("xnor2");
        xnor1(k, x, y) = popcount(binarize_A1(k, y) ^ binarize_B(k, x));
        xnor2(k, x, y) = popcount(binarize_A2(k, y) ^ binarize_B(k, x));

        output(x, y) = (-mean1(0)*((2 * cast<float>(sum(xnor1(bw.x, x, y), "accumulate1"))) - (64*bin_width))) + (-mean2(0)*((2 * cast<float>(sum(xnor2(bw.x, x, y), "accumulate2"))) - (64*bin_width)));

        if (!gpu) {
            /*binarize_A1.compute_root();
            binarize_A1.vectorize(x, 8);
            binarize_A2.compute_root();
            binarize_A2.vectorize(x, 8);
            mean1.compute_root();
            mean2.compute_root();
            binarize_B.compute_at(output, x);
            output.vectorize(x, 8);
            */
            mean1.compute_root();
            //mean1.vectorize(x, 8);
            mean2.compute_root();
            //mean2.vectorize(x, 8);
            binarize_A1.compute_at(output, y);
            binarize_A2.compute_at(output, y);
            //Var x_inner, x_outer;
            //output.split(x, x_outer, x_inner, 8);
            //output.vectorize(x_inner);
            //output.parallel(x_outer);
            //t.set_feature(Target::Profile);
        }
        output.compile_jit(t);
        
    }

    void realize(const float *A_array, float *out_array) {
        Buffer<float> outbuf = Buffer<float>(out_array, n, m);
        std::memcpy(A.get()->data(), A_array, m*p*sizeof(float));
        //std::memcpy(B.get()->data(), B_array, p*n/64*sizeof(int64_t));
        output.realize(outbuf);
    }
};

#endif
