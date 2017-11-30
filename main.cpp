#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <time.h>
#include "libsvm.h"
#include "dnn.h"

#define MPI_DEBUG false
#if MPI_DEBUG==true
#define MPI_DEBUG_CODE(rank) \
    if (dnn.world_rank == (rank)){ \
        int i = 0; \
        char hostname[256]; \
        gethostname(hostname, sizeof(hostname)); \
        printf("Partition %d: PID %d on %s ready for attach\n", dnn.world_rank, getpid(), hostname); \
        fflush(stdout); \
        while (0 == i) \
            sleep(5); \
    }
#else
#define MPI_DEBUG_CODE(rank)
#endif

#define NEWTON_MAX_ITER 10
#define MAX_LEN_FILENAME 128

//#define HEART_SCALE // Have to change the core number
#define HEART_SCALE_THREE // Have to change the core number
//#define POKER // Have to change the core number

#if defined(HEART_SCALE)
#define IS_INPUT_SPLIT false
#define DATA_NAME "heart_scale"
#define DATA_PATH "/home/loong/data/heart_scale"
#define NUM_INST 270
#define NUM_LABEL 2
#define NUM_FEAT 13
#define NUM_NEURON_EACH {13, 26, 26, 26, 26, 2}
#define NUM_SPLIT_EACH {2, 2, 2, 1, 1, 1}
#define NUM_LAYER 5
//#define NUM_NEURON_EACH {13, 26, 2}
//#define NUM_SPLIT_EACH {2, 1, 2}
//#define NUM_LAYER 2
#define LABEL_INIT 1

#elif defined(HEART_SCALE_THREE)
#define IS_INPUT_SPLIT false
#define DATA_NAME "heart_scale"
#define DATA_PATH "/home/loong/data/heart_scale_three"
#define NUM_INST 3
#define NUM_LABEL 3
#define NUM_FEAT 13
#define NUM_NEURON_EACH {13, 26, 3}
#define NUM_SPLIT_EACH {2, 3, 2}
#define NUM_LAYER 2
#define LABEL_INIT 1

#elif defined(POKER)
#define IS_INPUT_SPLIT false
#define DATA_PATH "/home/loong/data/poker"
#define DATA_NAME "poker"
#define NUM_INST 25010
#define NUM_LABEL 10
#define NUM_FEAT 10
#define NUM_NEURON_EACH {10, 26, 10}
#define NUM_SPLIT_EACH {2, 2, 1}
#define NUM_LAYER 2
#define LABEL_INIT 0
#endif

#define alpha0 1.0

int main(int argc, char** argv) {
    // Configure data settings
    char datafile[MAX_LEN_FILENAME] = DATA_PATH;
    INST_SZ numInst = NUM_INST;
    INST_SZ numClass = NUM_LABEL;
    FEAT_SZ numFeat = NUM_FEAT;
    int numNeuron[] = NUM_NEURON_EACH;
    int split[] = NUM_SPLIT_EACH;
    int numLayer = NUM_LAYER;
    char filename[MAX_LEN_FILENAME] = DATA_NAME;
    int labelInit = LABEL_INIT;

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

    // Configure NN settings
    DNN dnn(argc, argv);
    dnn.weightInit = &DNN::randomInit;
    dnn.activationFunc = new Activation*[numLayer]();
    for (int i=0; i<numLayer-1; ++i)
        dnn.activationFunc[i] = new Sigmoid();
    dnn.activationFunc[numLayer-1] = new Linear();
    //dnn.activationFunc[numLayer-1] = new Sigmoid();
    dnn.loss = &DNN::squareLoss;

    // Initial NN
    dnn.initial(argc, argv, numLayer, numNeuron, split);

    //MPI_DEBUG_CODE(0)
    MPI_DEBUG_CODE(0)
    //MPI_DEBUG_CODE(3)

    // Read data
    dnn.readInput(filename, datafile, numInst, numClass, numFeat, labelInit, IS_INPUT_SPLIT);

    dnn.activate();

    double *d;
    double alpha;
    bool isTrain = true;
    if (isTrain) {
        for (int i=0; i<NEWTON_MAX_ITER; ++i) {
            dnn.feedforward(isTrain);
            if (dnn.world_rank == 0) {
                printf("\niter %d\n", i);
                printf("[main] global_loss = %lf\n", dnn.getLoss());
            }
            dnn.backprop();
            dnn.calcJacobian();
            d = dnn.CG();
            alpha = dnn.line_search(alpha0, d);
            //dnn.update(0.001, d);
            if (dnn.world_rank == 0) {
                printf("[main] global_loss = %lf\n", dnn.getLoss());
            }

            delete[] d;
        }
    }
    else {
        dnn.feedforward(false);
    }

    dnn.finalize();
}

