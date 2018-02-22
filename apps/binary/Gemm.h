#ifndef GEMM
#define GEMM

#include "Halide.h"

using namespace Halide;

class Gemm{
    Buffer<float> A;
    Buffer<float> B;
    int n;
    int m;
    int k;
    Func output;
    Target t;
    Pipeline pline;
public:
    Gemm(float *B_data, int m, int n, int k, bool trans_A = false) :
    A(trans_A ? Buffer<float>(m, k) : Buffer<float>(k,m)),
    B(Buffer<float>(B_data, k, n)),
    n(n),
    m(m),
    k(k),
    t(get_host_target())
    {
        Var x("x"), y("y"), p("p");

        RDom r(0, k);

        Func A_in;
        if (trans_A) {
            A_in(x, y) = A(y, x);
        } else {
            A_in(x, y) = A(x, y);
        }

        output(x, y) = sum(A_in(r.x, y) * B(x, r.x), "accumulate");

        //output.vectorize(x, 16);
        //output.compile_jit();
        output.estimate(x, 0, n);
        output.estimate(y, 0, m);
        pline = Pipeline(output);
        pline.auto_schedule(t);
        pline.compile_jit(t);
    }

    void realize(float *A_array, float *out_array) {
        Buffer<float> outbuf = Buffer<float>(out_array, n, m);
        std::memcpy(A.get()->data(), A_array, m*k*sizeof(float));
        pline.realize(outbuf);
    }
};

#endif
