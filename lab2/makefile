CXX       = nvcc
ECHO      = /bin/echo
CFLAGS    = --std=c++11 -arch sm_30 -O2 -I../utils

all: couting.o main.cu 
	@$(ECHO) $(CXX) $(CFLAGS) *.o main.cu -o main
	@$(CXX) $(CFLAGS) *.o main.cu -o main

couting.o: counting.cu
	@$(CXX) $(CFLAGS) -c counting.cu