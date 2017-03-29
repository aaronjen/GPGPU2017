CXX       = nvcc
ECHO      = /bin/echo
CFLAGS    = -O3 -std=c++11

all: PerlinNoise.o lab1.o main.cu 
	@$(ECHO) $(CXX) $(CFLAGS) *.o main.cu -o main
	@$(CXX) $(CFLAGS) *.o main.cu -o main

PerlinNoise.o: PerlinNoise.h PerlinNoise.cpp
	@$(CXX) $(CFLAGS) -c PerlinNoise.cpp

lab1.o : lab1.h lab1.cu particle.h
	@$(CXX) $(CFLAGS) -c lab1.cu

run:
	./main
	rm -f result.mp4
	ffmpeg -i result.y4m result.mp4

clean:
	@rm -f main *.o

