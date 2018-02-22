//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "Halide.h"
#include "HalideRuntime.h"
#include <stdio.h>

using namespace Halide;
int main(int argc, char **argv) {
    ImageParam A(type_of<float>(), 2, "A");
    ImageParam B(type_of<float>(), 2, "B");

    Expr K = A.width();
    
    Var x("x"), y("y"), c("c"), f("f"), k("k");
    
    Target target;
    target = get_host_target();
    //target.os = Target::Windows;
    //target.arch = Target::X86;
    //target.bits = 64;

    //std::vector<Target::Feature> profile_features;
    //profile_features.push_back(Target::AVX);
    //profile_features.push_back(Target::SSE41);
    //profile_features.push_back(Target::Profile);
    //target.set_features(profile_features);

    Func AIn("AIn");
    Func BIn("BIn");
    AIn(x, y) = BoundaryConditions::constant_exterior(A, 0)(x, y);
    // transpose B so it can be binarized easily
    BIn(x, y) = BoundaryConditions::constant_exterior(B, 1)(y, x);

    Func binarize_A("binarize_A");
    RDom r(0, 64);

    binarize_A(x, y) = sum(select(AIn(64*x + r, y) > 0, cast<int64_t>(1) << r.x, cast<int64_t>(0)), "compress_A"); 

    Func binarize_B("binarize_B");

    binarize_B(x, y) = sum(select(BIn(64*x + r, y) > 0, cast<int64_t>(1) << r.x, cast<int64_t>(0)), "compress_B"); 

    Func xnor("xnor");
    xnor(k, x, y) = popcount(binarize_A(k, y) ^ binarize_B(k, x));
    //xnor(k, x, y) = popcount(binarize_weights(k, y));

    Func output("output");
    Expr bin_width = ((K-1)/64) + 1;
    RDom bw(0, bin_width);
    output(x, y) = -((2 * cast<float>(sum(xnor(bw.x, x, y), "accumulate"))) - (64*bin_width));

    // scheduling
       
    Var x_inner, x_outer, y_inner, y_outer;
    //binarize_A.compute_root();
    //binarize_A.vectorize(x, 8);
    //binarize_A.parallel(y, 4);
    //binarize_B.compute_root();
    //binarize_B.vectorize(x, 8);
    //binarize_B.parallel(y, 8);
    //binarize_B.compute_root();
    //binarize_B.vectorize(x, 8);
    //binarize_B.parallel(f, 8);
    output.reorder(y, x);
    //binarize_input.compute_root();
    //output.unroll(y, 4);
    //binarize_B.compute_root();
    //binarize_B.vectorize(x, 4);
    binarize_A.compute_root();
    binarize_A.vectorize(x, 4);
    //binarize_B.parallel(y, 8);
    //binarize_A.compute_at(output, y);
    output.vectorize(x, 4);
    output.parallel(y, 8);
    //binarize_A.compute_at(output, x);
    binarize_B.compute_at(output, x);
    
    std::vector<Argument> args = {A, B};
    output.compile_to_static_library("halide_bitgemm", args, "halide_bitgemm", target);
    //output.compile_to_file("halide_convolve", args, "halide_convolve", target);
    return 0; 
} 
