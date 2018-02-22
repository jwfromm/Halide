#ifndef BINDEPTH
#define BINDEPTH

#include "Halide.h"

using namespace Halide;

class BinDepth {
    Buffer<float> input;
    Buffer<int64_t> weights;
    Func output;
    Target t;
    int filters;
    int size;
    int stride;
    int pad;
    int w;
    int h;
    int channels;
    int binchannels;
    int out_x;
    int out_y;
    int bin_adjust;
public:
    BinDepth(int64_t *W, int w, int h, int channels, int filters, int size, int stride, int pad) :
    input(Buffer<float>(w,h,channels)),
    weights(Buffer<int64_t>(W, size, size, channels/64, filters)),
    w(w),
    h(h),
    channels(channels),
    binchannels(channels/64),
    size(size),
    filters(filters),
    stride(stride),
    pad(pad),
    bin_adjust(size*size*channels),
    out_x(!pad ? (w - size) / stride + 1 : (w - 1)/stride + 1),
    out_y(!pad ? (h - size) / stride + 1 : (h - 1)/stride + 1),
    t(get_host_target())
    {
        Var x("x"), y("y"), c("c"), k("k");
        Func clamped;
        clamped(x, y, c) = BoundaryConditions::constant_exterior(input, 0)(x, y, c);
        Func binclamped;
        RDom b(0, 64);
        binclamped(x, y, c) = sum(select(clamped(x, y, 64*c + b) > 0, cast<int64_t>(1) << b, cast<int64_t>(0)), "binarize_input");
        RDom r(0, size, 0, size, 0, binchannels);
        //output(x, y, k) = sum(weights(r.x, r.y, r.z, k) * binclamped(x * stride + r.x - pad, y*stride + r.y - pad, r.z));
        output(x, y, k) = -cast<float>((2 * (sum(popcount(weights(r.x, r.y, r.z, k) ^ binclamped(x * stride + r.x - pad, y*stride + r.y - pad, r.z))))) - bin_adjust);

        Var x_outer, x_inner, y_outer, y_inner, tile_index;
        output.tile(x, y, x_outer, y_outer, x_inner, y_inner, 16, 16);
        output.reorder(x_inner, y_inner, k, x_outer,y_outer);
        output.fuse(x_outer, y_outer, tile_index);
        output.parallel(tile_index);
        output.vectorize(x_inner);
        
        binclamped.store_root().compute_root();
        t.set_feature(Target::Profile);
        output.compile_jit(t);
    }
    void realize(float *in_array, float *out_array) {
        Buffer<float> outbuf = Buffer<float>(out_array, out_x, out_y, filters);
        std::memcpy(input.get()->data(), in_array, w*h*channels*sizeof(float));
        output.realize(outbuf);
    }
};
#endif
