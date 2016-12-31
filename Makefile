MPICC=mpicc
CFLAGS=-O2 #-c -Wall
LDFLAGS=-lstdc++
EXEC_FLAGS=-np 4
SRC=main.cpp
OBJ=$(SRC:.cpp=.o)
TARGET=./main

.PHONY: clean all run

all: $(SRC) $(TARGET)
    
$(TARGET): $(SRC) 
	$(MPICC) $< $(CFLAGS) $(LDFLAGS) -o $(TARGET)

run:
	mpiexec $(EXEC_FLAGS) $(TARGET)

clean:
	rm -rf $(OBJ) $(TARGET)
