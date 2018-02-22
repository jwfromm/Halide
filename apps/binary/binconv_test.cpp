#include "Halide.h"
#include <stdio.h>
#include <chrono>
#include "BinConvolve.h"

using namespace std;
using ns = chrono::nanoseconds;
using get_time = chrono::steady_clock;
using namespace Halide;

int main(int argc, char **argv) {
    //HalideExecute tester = new HalideExecute();
    BinConvolve test = BinConvolve(10, 10, 64, 1, 3, 1, 1, 0);
    float in_test[3000] = {};
    float out_test[3000] = {};
    float weight_test[3000] = {};
    test.realize(in_test, weight_test, out_test);
    return 0;
}
