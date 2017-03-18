#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include "libsvm.h"
#include "dnn.h"

#define MAX_ITER 200
#define MAX_LEN_FILENAME 128

#define HEART_SCALE

#ifdef HEART_SCALE
#define IS_INPUT_SPLIT false
#define DATA_PATH "/home/loong/data/heart_scale"
#define DATA_NAME "heart_scale"
#define NUM_INST 270
#define NUM_LABEL 2
#define NUM_FEAT 13
#define NUM_NEURON_EACH {13, 26, 2}
#define NUM_SPLIT_EACH {2, 4, 1}
#define NUM_LAYER 2
#endif

#ifdef POKER
#define IS_INPUT_SPLIT false
#define DATA_PATH "/home/loong/data/poker"
#define DATA_NAME "poker"
#define NUM_INST 25010
#define NUM_LABEL 10
#define NUM_FEAT 10
#define NUM_NEURON_EACH {10, 26, 2}
#define NUM_SPLIT_EACH {2, 4, 1}
#define NUM_LAYER 2
#endif

int main(int argc, char** argv) {
    int rank; // Get the rank of the process
    int size; // Get the number of processes
    int name_len;
    char processor_name[MPI_MAX_PROCESSOR_NAME]; // Get the name of the processor

    // Read Data
    char datafile[MAX_LEN_FILENAME] = DATA_PATH;
    INST_SZ numInst = NUM_INST;
    INST_SZ numClass = NUM_LABEL;
    FEAT_SZ numFeat = NUM_FEAT;

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
    int numNeuron[] = NUM_NEURON_EACH;
    int split[] = NUM_SPLIT_EACH;
    int numLayer = NUM_LAYER;
    char filename[MAX_LEN_FILENAME] = DATA_NAME;
    
    DNN dnn;
    dnn.initial(argc, argv, numLayer, numNeuron, split);
    dnn.readInput(filename, datafile, numInst, numClass, numFeat, IS_INPUT_SPLIT);
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

    dnn.finalize();
    /*
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
    */
}

