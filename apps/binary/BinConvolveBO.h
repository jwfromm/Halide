#ifndef BINCONVOLVEBO
#define BINCONVOLVEBO

#include "Halide.h"

using namespace Halide;

class BinConvolveBO {
    Buffer<float> input;
    Buffer<float> weights;
    int filters;
    int size;
    int stride;
    bool pad;
    int w;
    int h;
    int channels;
    int out_x;
    int out_y;
    int bin_width;
    Func output;
    Func binarize_output;
    Target t;
public:
    BinConvolveBO(int w, int h, int channels, int filters, int size, int stride, bool pad, bool gpu = false) :
    input(Buffer<float>(w,h,channels)),
    weights(Buffer<float>(size*size*channels, filters)),
    filters(filters),
    size(size),
    stride(stride),
    pad(pad),
    w(w),
    h(h),
    channels(channels),
    out_x(!pad ? (w - size) / stride + 1 : (w - 1)/stride + 1),
    out_y(!pad ? (h - size) / stride + 1 : (h - 1)/stride + 1),
    bin_width((size*size*channels - 1)/64 + 1),
    t(get_host_target())
    {
        Var x("x"), y("y"), c("c"), f("f"), k("k");

        Func Input("Input");
        Input(x, y, c) = BoundaryConditions::constant_exterior(input, 0)(x, y, c);

        Func binarize_input("binarize_input"), bit_mask("bit_mask"), mask_count("mask_count");
        RDom r(0, 64);

        Expr w_offset = (y % out_x)*stride;
        Expr h_offset = ((y / out_x) % out_y) * stride;

        Expr im_row = h_offset + ((64*x + r.x)/size) % size - select(pad, size/2, 0); 
        Expr im_col = w_offset + (64*x + r.x) % size - select(pad, size/2, 0); 
        Expr im_chan = (64*x + r.x) / size / size;

        RDom bw(0, bin_width);
        
        binarize_input(x, y) = sum(select(Input(im_col, im_row, im_chan) > 0, cast<int64_t>(1) << r.x, cast<int64_t>(0)), "compress_inputs"); 
        bit_mask(x, y) = sum(select((im_row < 0 || im_col < 0 || im_row >= input.height() || im_col >= input.width()), cast<int64_t>(0) << r.x, cast<int64_t>(1) << r.x), "make_bitmask");
        mask_count(y) = sum(popcount(~bit_mask(bw.x, y)));

        Func binarize_weights("binarize_weights");
        RDom n(0, weights.width());
        binarize_weights(x, f) = sum(select(weights(64*x + r.x, f) > 0, (cast<int64_t>(1)) << r.x, cast<int64_t>(0)), "compress_weights");

        Func xnor("xnor");
        xnor(k, x, y) = (popcount(bit_mask(k, x) & (binarize_weights(k, y) ^ binarize_input(k, x))));

        output(x, y) = -((2 * cast<float>(sum(xnor(bw.x, x, y), "accumulate"))) - (64*bin_width) + mask_count(x));
        binarize_output(x, y) = sum(select(output(64*x + r.x, y) > 0, (cast<int64_t>(1)) << r.x, cast<int64_t>(0)), "compress_outputs");

        binarize_weights.compute_root();
        binarize_weights.vectorize(x, 8);
        binarize_weights.parallel(f, 8);
        binarize_output.reorder(y, x);
        binarize_output.vectorize(y, 8);
        binarize_output.parallel(x, 8);
        //binarize_input.compute_at(output, x);
        //bit_mask.compute_at(output, x);
        binarize_input.compute_root();
        bit_mask.compute_root();
        output.compile_jit(t);
        
    }

    void realize(float *in_array, float *weight_array, int64_t *out_array) {
        Buffer<int64_t> outbuf = Buffer<int64_t>(out_array, (out_x*out_y+1)/64 - 1, filters);
        std::memcpy(input.get()->data(), in_array, w*h*channels*sizeof(float));
        std::memcpy(weights.get()->data(), weight_array, size*size*channels*filters*sizeof(float));
        binarize_output.realize(outbuf);
    }
};

#endif
