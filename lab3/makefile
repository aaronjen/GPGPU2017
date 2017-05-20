CXX       = nvcc
ECHO      = /bin/echo
CFLAGS    = --std=c++11 -arch sm_30 -O2 -I../utils

all: main.cu lab3.cu
	@$(ECHO) $(CXX) $(CFLAGS) lab3.cu main.cu pgm.cpp -o main
	@$(CXX) $(CFLAGS) lab3.cu main.cu pgm.cpp -o main
	./main img_background.ppm img_target.ppm img_mask.pgm 130 600 output.ppm
