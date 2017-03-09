#ifndef _DNN_H_
#define _DNN_H_

#include "libsvm.h"
#include <mpi.h>

class DNN;
typedef void (DNN::*fpWeightInit)();
typedef double (DNN::*fpActvFunc)(double *);
typedef double (DNN::*fpLoss)(double *);
typedef double floatX;

class DNN {
    // Initialize the MPI environment
    private:
        int numLayer;
        int curLayer;
        int *numNeuron; // 28-300-300-1
        int *split; // 2-2-1-1
        int *layerId;
        int *masterId;
        int prevSplitId;
        int nextSplitId;
        int *numPartition;
        int *numNeurInSet;
        floatX *weight; // 28*300, 300*300, 300*1
        floatX *biases; // 300, 300, 1
        floatX *X;
        int *Y; // one-hot for multiclass
        double linear(double *x);
        double sigmoid(double *x);
        double relu(double *x);
        double tanh(double *x);
        double softmax(double *x);
        double logLoss(double *x);
        double squareLoss(double *x);
        double l1Loss(double *x);
        void allocWeight();
        void allocBiases();
        void randomInit();
        void sparseInit();
        int world_rank; // Get the rank of the process
        int world_size; // Get the number of processes
        void initMPI(int, char**);
        void initLayerId();
        void initSplitId();
        void initNeuronSet();
        void formMPIGroup();
        MPI_Comm recvComm;  // recv from previous layer together with split[n] partitions
        MPI_Comm bcastComm;  // intercomm of recv broadcast from split[n-1] partitions
        MPI_Comm prevBcastComm;  // intercomm of recv broadcast from split[n-1] partitions
        MPI_Comm nextBcastComm;  // intercomm of send broadcast to split[n+1] partitions
        MPI_Comm reduceComm;  // split[n-1] partitions do reduce
        MPI_Group recvGrp;
        MPI_Group bcastGrp;
        MPI_Group reduceGrp;

    public:
        virtual void initial(int argc, char **argv, const int numLayer, int *numNeuron, int *split);
        void readInput(LIBSVM data);
        void readWeightFromFile(char *filename);
        void readBiasesFromFile();
        void readWeight();
        void readBiases();
        void finalize();
        //void (DNN::*weightInit)();
        fpWeightInit weightInit;
        fpActvFunc *activationFunc; // Function pointer array
        fpLoss loss;
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
