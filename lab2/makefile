CXX       = nvcc
ECHO      = /bin/echo
CFLAGS    = --std=c++11 -arch sm_30 -O2 -I../utils

all: main.cu counting.cu
	@$(ECHO) $(CXX) $(CFLAGS) counting.cu main.cu -o main
	@$(CXX) $(CFLAGS) counting.cu main.cu -o main
