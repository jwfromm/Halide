#ifndef TESTCONV
#define TESTCONV

#include "Halide.h"

using namespace Halide;

class TestConv {
    Buffer<float> input;
    Buffer<float> weights;
    Func output;
    Target t;
    int filters;
    int size;
    int stride;
    int pad;
    int w;
    int h;
    int channels;
    int out_x;
    int out_y;
public:
    TestConv(float *W, int w, int h, int channels, int filters, int size, int stride, int pad) :
    input(Buffer<float>(w,h,channels)),
    weights(Buffer<float>(W, size, size, channels, filters)),
    w(w),
    h(h),
    channels(channels),
    size(size),
    filters(filters),
    stride(stride),
    pad(pad),
    out_x(!pad ? (w - size) / stride + 1 : (w - 1)/stride + 1),
    out_y(!pad ? (h - size) / stride + 1 : (h - 1)/stride + 1),
    t(get_host_target())
    {
        Var x("x"), y("y"), c("c"), k("k");
        Func clamped;
        clamped(x, y, c) = BoundaryConditions::constant_exterior(input, 0)(x, y, c);
        RDom r(0, size, 0, size, 0, channels);
        output(x, y, k) = sum(weights(r.x, r.y, r.z, k) * clamped(x * stride + r.x - pad, y*stride + r.y - pad, r.z));

        output.compute_root();
        output.parallel(k, 8);
        output.vectorize(x, 16);
        clamped.store_root().compute_root();
        t.set_feature(Target::Profile);
        output.compile_jit(t);
    }
    void realize(float *in_array, float *out_array) {
        printf("%d", (w - size)/stride + 1);
        Buffer<float> outbuf = Buffer<float>(out_array, out_x, out_y, filters);
        std::memcpy(input.get()->data(), in_array, w*h*channels*sizeof(float));
        output.realize(outbuf);
    }
};
#endif
