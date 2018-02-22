valgrind --tool=callgrind ./binarization_test
kcachegrind callgrind.out*
rm callgrind.out*
