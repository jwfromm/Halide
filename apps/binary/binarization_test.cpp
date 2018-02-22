#include "Halide.h"
#include <stdio.h>
#include <chrono>

using namespace std;
using ns = chrono::nanoseconds;
using get_time = chrono::steady_clock;
using namespace Halide;

class HalideExecute {
    ImageParam input;
    Func output;
    Target t;
public:
    HalideExecute() {
        input = ImageParam(type_of<float>(), 2);
        Var x("x"), y("y");
        output(x,y) = (input(x, y) + input(x, y+1) + input(x + 1, y), input(x+1, y+1))/4;    
        output.trace_stores();
        output.compile_jit();
    }
    
    void compute(float *in_array, int x, int y, float* out_array) {
        Buffer<float> inbuf = Buffer<float>(in_array, x+1, y+1);
        in_array[0] = 10;
        out_array[0] = 3;
        memcpy(inbuf.get()->data(), out_array, 4);
        cout<<"Value:" <<inbuf.get()->data()[0]<<endl;
        //cout<<"in_array:"<<in_array[0]<<endl;
        Buffer<float> outbuf = Buffer<float>(out_array, x, y);
        input.set(inbuf);
        output.realize(outbuf);
    }
};

int main(int argc, char **argv) {
    Var x("x"), y("y");
    
    //Func producer("producer_default"), consumer("consumer_default");
    Func consumer;
    
    //producer(x, y) = sin(x*y);
    ImageParam producer(type_of<float>(), 2);
    
    consumer(x, y) = (producer(x, y) + producer(x, y+1) + producer(x + 1, y), producer(x+1, y+1))/4;
    
    consumer.trace_stores();

    Target t = get_host_target();
    t.set_feature(Target::Profile);
    (consumer.compile_jit(t));

    float test_vals[9] = {};
    float test_out[4] = {};

    Buffer<float> inbuf = Buffer<float>(test_vals, 3, 3);
    producer.set(inbuf);

    Buffer<float> output(2, 2);
    //consumer.realize(output);

    //HalideExecute tester = new HalideExecute();
    HalideExecute tester;
    for (int i = 0; i < 1; i ++) {
        auto start = get_time::now();
        tester.compute(test_vals, 2, 2, test_out);
        auto end = get_time::now();
        auto diff = end - start;
        cout<<"Time for round "<<i<<" is "<<chrono::duration_cast<ns>(diff).count()<<" ns "<<endl;
    }

    return 0;
}
