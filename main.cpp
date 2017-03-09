#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include "libsvm.h"
#include "dnn.h"

#define MAX_ITER 200

int main(int argc, char** argv) {
    int rank; // Get the rank of the process
    int size; // Get the number of processes
    int name_len;
    char processor_name[MPI_MAX_PROCESSOR_NAME]; // Get the name of the processor

    // Initialize the MPI environment
    /*
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(processor_name, &name_len);
    */

    // Read Data
    char datafile[50] = "/home/loong/data/poker";
    INST_SZ numInst = 25010;
    INST_SZ numLabel = 10;
    FEAT_SZ numFeat = 10;
    LIBSVM data(datafile, numInst, numLabel, numFeat);

    /*
    printf("data.label: \n");
    for (int i=0; i<data.numInst; ++i) {
        printf("%d ", data.label[i]);
        for (int j=0; j<data.numFeat; ++j) {
            printf("%d:", data.idx[i*data.numFeat + j]);
            printf("%d", data.feat[i*data.numFeat + j]);
            if (j!=data.numFeat - 1) printf(" ");
        }
        printf("\n");
    }
    */

    /*
    int numNeuron[] = {13, 26, 26, 2};
    int split[] = {2, 2, 2, 1};
    int numLayer = 3;
    */
    int numNeuron[] = {13, 26, 2};
    int split[] = {2, 4, 1};
    int numLayer = 2;
    DNN dnn;
    dnn.initial(argc, argv, numLayer, numNeuron, split);
    dnn.readInput(data);
    dnn.readWeight();
    //dnn.DNN::*weightInit();
    //dnn.activationFunc[2]();

    //initial(weight, biases);

    //dnn.feedforward();
    for (int i=0; i<MAX_ITER; ++i) {
        /*
           dnn.feedforward();
        //dnn.calcGradient();
        dnn.backforward();
        dnn.calcJacobian();
        dnn.calcJBJv();
        dnn.CG();
        dnn.update();
        */
    }

    //printf("Hello world from processor %s, rank %d"
    //        " out of %d processors\n", processor_name, rank, size);

    dnn.finalize();
    printf("Hello world from processor");
    {
        int i = 0;
        char hostname[256];
        gethostname(hostname, sizeof(hostname));
        printf("PID %d on %s ready for attach\n", getpid(), hostname);
        fflush(stdout);
        while (0 == i)
            sleep(5);
    }
}

