CXX       = nvcc
ECHO      = /bin/echo
CFLAGS    = -O3 -std=c++11 -I../utils

all: couting.o main.cu 
	@$(ECHO) $(CXX) $(CFLAGS) *.o main.cu -o main
	@$(CXX) $(CFLAGS) *.o main.cu -o main

couting.o: counting.cu
	@$(CXX) $(CFLAGS) -c counting.cu