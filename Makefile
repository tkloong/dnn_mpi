MPICC=mpicc
CFLAGS=-O2 -g #-c -Wall
LDFLAGS=-lstdc++
EXEC_FLAGS=-np 12
SRC=main.cpp
OBJ=$(SRC:.cpp=.o)
TARGET=./main

.PHONY: clean all run

all: $(SRC) dnn.cpp libsvm.h $(TARGET)
    
$(TARGET): $(SRC) dnn.cpp libsvm.h
	$(MPICC) $< dnn.cpp $(CFLAGS) $(LDFLAGS) -o $(TARGET)

run:
	mpiexec $(EXEC_FLAGS) $(TARGET)

clean:
	rm -rf $(OBJ) $(TARGET)
