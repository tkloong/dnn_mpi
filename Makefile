MPICC=mpicc
#CFLAGS=-O2 -g #-c -Wall
CFLAGS=-g
LDFLAGS=-lstdc++ -lm -lopenblas -lpthread -lgfortran
EXEC_FLAGS=-np 12  # HEART_SCALE
#EXEC_FLAGS=-np 6  	# POKER
SRC=main.cpp
OBJ=$(SRC:.cpp=.o)
TARGET=./main

.PHONY: clean all run

all: $(SRC) dnn.h dnn.cpp libsvm.h libsvm.cpp $(TARGET)
    
$(TARGET): $(SRC) dnn.h dnn.cpp libsvm.h libsvm.cpp
	$(MPICC) $< dnn.cpp libsvm.cpp $(CFLAGS) -o $(TARGET) $(LDFLAGS)

run:
	mpiexec $(EXEC_FLAGS) $(TARGET)

gdb:
	mpiexec -n 1 gdb ./main : -n 11 ./main

clean:
	rm -rf $(OBJ) $(TARGET)
