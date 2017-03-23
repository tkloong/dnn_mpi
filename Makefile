MPICC=mpicc
#CFLAGS=-O2 -g #-c -Wall
CFLAGS=-g
LDFLAGS=-lstdc++ -lopenblas -lpthread -lgfortran
EXEC_FLAGS=-np 12
SRC=main.cpp
OBJ=$(SRC:.cpp=.o)
TARGET=./main

.PHONY: clean all run

all: $(SRC) dnn.cpp libsvm.h libsvm.cpp $(TARGET)
    
$(TARGET): $(SRC) dnn.cpp libsvm.h libsvm.cpp
	$(MPICC) $< dnn.cpp libsvm.cpp $(CFLAGS) -o $(TARGET) $(LDFLAGS)

run:
	mpiexec $(EXEC_FLAGS) $(TARGET)

gdb:
	mpiexec -n 1 gdb ./main : -n 11 ./main

clean:
	rm -rf $(OBJ) $(TARGET)
