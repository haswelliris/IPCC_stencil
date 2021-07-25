# CXX=icpc
# CXXFLAGS=-Iinclude/ -qopenmp -Ofast -march=skylake-avx512 -g
CXX=g++
CXXFLAGS=-Iinclude/ -fopenmp -Ofast -mavx2 -mfma -g
LDFLAGS=lib/libpng16.a lib/libz.a

OBJECTS=src/main.o src/image.o src/stencil.o

stencil: $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o bin/stencil $(OBJECTS) $(LDFLAGS)

all:	stencil

run:	all
	bin/stencil IPCC.png

clean:
	rm -f $(OBJECTS) bin/stencil output.png src/*~ *~
