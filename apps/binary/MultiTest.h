#ifndef MULTITEST
#define MULTITEST

#include "Halide.h"

using namespace Halide;

class MultiTest {
    Buffer<float> A;
    int n;
    int bits;
    int bin_width;
    Func output;
    Target t;
public:
    MultiTest(int n, int bits, bool gpu = false) :
    A(Buffer<float>(n)),
    n(n),
    bits(bits),
    bin_width((n - 1)/64 + 1),
    t(get_host_target())
    {
        Var x("x"), y("y"), k("k");

        Func binarize_A("binarize_A"), carry_over("carry_over");
        RDom r(0, 64);
        RDom bw(0, bin_width);
        RDom rn(0, n);
        RDom bitdom(1, bits);
        
        //binarize_A(x, y) = sum(select(A(64*x + r, y) > 0, cast<int64_t>(1) << r.x, cast<int64_t>(0)), "compress_A"); 
        carry_over(x, y) = A(x);
        //carry_over(x, bitdom) = carry_over(x, bitdom-1) + select(carry_over(x, bitdom-1) > 0, -sum(rn, abs(carry_over(rn, bitdom-1))), sum(rn, abs(carry_over(rn, bitdom-1))));

        output(x) = carry_over(x, 0);

        output.compile_jit(t);
        
    }

    void realize(float *A_array, float *out_array) {
        Buffer<float> outbuf = Buffer<float>(out_array, n);
        std::memcpy(A.get()->data(), A_array, n*sizeof(float));
        output.realize(outbuf);
    }
};

#endif
