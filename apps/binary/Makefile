CXX=g++
#CXXFLAGS=-std=c++11 -Ofast
CXXFLAGS=-std=c++11 -g

all: convolve_test gemm_test conv_test

convolve_test: convolve_test.cpp BinConvolve.h BinGemm.h MultiTest.h BinGemm2A1B.h BinConvolveBO.h BinConvolveBIBO.h BinConvolveBI.h BinConv1A1W.h TestConv.h BinDepth.h BinGemm1A1B.h
	$(CXX) $(CXXFLAGS) -I/opt/OpenBLAS/include/ -L/opt/OpenBLAS/lib/ convolve_test.cpp -lHalide -lopenblas -o convolve_test

gemm_test: gemm_test.cpp BinGemm1A1B.h Gemm.h
	$(CXX) gemm_test.cpp -g -I/opt/OpenBLAS/include/ -L/opt/OpenBLAS/lib/ -I ../../include -L ../../bin -lHalide -lpthread -ldl -lopenblas -o gemm_test -std=c++11	    	

conv_test: conv_test.cpp BinDepth.h BinGemm1A1B.h
	$(CXX) conv_test.cpp -g -I/opt/OpenBLAS/include/ -L/opt/OpenBLAS/lib/ -I ../../include -L ../../bin -lHalide -lpthread -ldl -lopenblas -o conv_test -std=c++11	    	

