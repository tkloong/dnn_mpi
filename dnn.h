#ifndef _DNN_H_
#define _DNN_H_

#include "libsvm.h"
#include "libsvm.hpp"
#include <mpi.h>
#include <stdlib.h>

class DNN;
typedef void (DNN::*fpWeightInit)();
typedef double (DNN::*fpActvFunc)(double *);
typedef double (DNN::*fpLoss)(double *);
typedef double floatX;

class DNN {
    private:
        int numLayer;       // Total number of layer for this NN.
        int curLayer;       // Layer Id in this partition.
        int *numNeuron;     // Array of number of neuron for this NN structure. E.g. 28-300-300-1.
        int *split;         // Split structure for this NN. E.g. 2-2-1-1.
        int *layerId;       // Array of layer Id for each partition.
        int *masterId;      // Array of master Id for each layer.
        int prevSplitId;    // Previous split id in this partition.
        int nextSplitId;    // Next split id in this partition.
        int prevEle;        // First dimension of weight matrix.
        int nextEle;        // Second dimension of weight matrix.
        int *numPartition;  // Total number of partitions (Not used)
        int *numNeurInSet;  // Array of floor(number of neurons in partitions for each layer).
        floatX *weight;     // Weight matrix in this partition. E.g. 28*300, 300*300, 300*1.
        floatX *biases;     // Biases in this partition. E.g. 300, 300, 1.
        floatX *X;          // Array of input feature
        int *Y;             // Array of one-hot label for multiclass
        int world_rank;     // Rank of the process
        int world_size;     // Number of the processes
        MPI_Comm recvComm;  // recv from previous layer together with split[n] partitions
        MPI_Comm bcastComm; // intercomm of recv broadcast from split[n-1] partitions
        MPI_Comm prevBcastComm; // intercomm of recv broadcast from split[n-1] partitions
        MPI_Comm nextBcastComm; // intercomm of send broadcast to split[n+1] partitions
        MPI_Comm reduceComm;    // split[n-1] partitions do reduce
        MPI_Group recvGrp;
        MPI_Group bcastGrp;
        MPI_Group reduceGrp;
        void allocWeight();
        void allocBiases();
        void initMPI(int, char**);
        void initLayerId();
        void initSplitId();
        void initNeuronSet();
        void formMPIGroup();
        void NOT_DEF();
        LIBSVM *data;
        //void NOT_DEF(double *);

    public:
        DNN();
        virtual void initial(int argc, char **argv, const int numLayer, int *numNeuron, int *split);
        void readInput(char *prefixFilename, char *datafile=NULL, INST_SZ numInst=0, INST_SZ numClass=0, FEAT_SZ numFeat=0, bool isFileExist=true);
        void readWeightFromFile(char *filename);
        void readBiasesFromFile();
        void readWeight();
        void readBiases();
        void finalize();
        //void (DNN::*weightInit)();
        fpWeightInit weightInit;
        fpActvFunc *activationFunc; // Function pointer array
        fpLoss loss;
        void randomInit();
        void sparseInit();
        double linear(double *x);
        double sigmoid(double *x);
        double relu(double *x);
        double tanh(double *x);
        double softmax(double *x);
        double logLoss(double *x);
        double squareLoss(double *x);
        double l1Loss(double *x);
        //double (*activationFunc[])(double *);
        void feedforward();
        //void calcGradient();
        void backforward();
        void calcJacobian();
        void calcJBJv();
        void CG();
        void update();
};

#endif
