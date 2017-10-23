#ifndef _DNN_H_
#define _DNN_H_

#include "libsvm.h"
#include "libsvm.hpp"
//extern "C" {
//#include <cblas.h>
//}
#include "mkl.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>

#define SIGNAL_MAX_LEN 128

class Activation;
class Sigmoid;
class Linear;
class DNN;
typedef void (DNN::*fpWeightInit)();
typedef double (DNN::*fpLoss)(LABEL *lable, double *x, int *inst, int *unit, int *startLbl, int *stopLbl);
typedef double floatX;
typedef struct {
    int rank;
    char msg[SIGNAL_MAX_LEN];
} Signal;

class DNN {
    private:
        int numLayer;       // Total number of layer for this NN, excluding input layer.
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
        floatX *dXidZ;      // Gradient of the units in output layer
        //floatX *dXids;      // Gradient of the units in output layer
        floatX *dXidW;      // Gradient of the units in output layer
        floatX *dXidb;      // Gradient of the units in output layer
        double *z;			// instance rows by n_m columns
        double *zPrev;		// instance rows by n_{m-1} columns
        floatX *X;          // Array of input feature
        int *Y;             // Array of one-hot label for multiclass
        int batchSize;      // For pipeline in function value evaluation
        double C;           // Regularization coefficient
        int world_rank;     // Rank of the process
        int world_size;     // Number of the processes
        MPI_Comm recvComm;  // recv from previous layer together with split[n] partitions
        //MPI_Comm bcastComm; // intercomm of recv broadcast from split[n-1] partitions
        MPI_Comm prevBcastComm; // intercomm of recv broadcast from split[n-1] partitions
        MPI_Comm nextBcastComm; // intercomm of send broadcast to split[n+1] partitions
        MPI_Comm reduceComm;    // split[n-1] partitions do reduce
        MPI_Comm funcValComm; // intercomm to calculate the function value
        MPI_Group recvGrp;
        MPI_Group bcastGrp;
        MPI_Group reduceGrp;
        void allocWeight();
        void allocBiases();
        void allocGradient();
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
        MPI_Datatype Mpi_signal;
        Signal sendBuf;
        Signal *recvBuf;
        void CreateSignalType (Signal *signal);
        virtual void initial(int argc, char **argv, const int numLayer, int *numNeuron, int *split);
        void readInput(char *prefixFilename, char *datafile=NULL, INST_SZ numInst=0, INST_SZ numClass=0, FEAT_SZ numFeat=0, int labelInit=-1, bool isFileExist=true);
        void readWeightFromFile(char *filename);
        void readBiasesFromFile();
        void readWeight();
        void readBiases();
        void finalize();
        //void (DNN::*weightInit)();
        fpWeightInit weightInit;
        Activation **activationFunc;
        fpLoss loss;
        void randomInit();
        void sparseInit();
        double linear(double *x, int len);
        double sigmoid(double *x, int len);
        double relu(double *x, int len);
        double tanh(double *x, int len);
        double softmax(LABEL *lable, double *x, int *inst, int *unit, int *startLbl, int *stopLbl);
        double logLoss(LABEL *lable, double *x, int *inst, int *unit, int *startLbl, int *stopLbl);
        double squareLoss(LABEL *lable, double *x, int *inst, int *unit, int *startLbl, int *stopLbl);
        double squareLossCalc(LABEL *label, double *x, int *inst, int *unit, int *startLbl, int *stopLbl);
        double l1Loss(LABEL *lable, double *x, int *inst, int *unit, int *startLbl, int *stopLbl);
        //double (*activationFunc[])(double *);
        void setInstBatch(int batchSize);
        void feedforward(bool isTrain);
        //void calcGradient();
        void backprop();
        void calcJacobian();
        void calcJBJv();
        void CG();
        void update();
};

class Activation
{
    public:
        virtual floatX* grad(double *ptr, int len) {};
        virtual double calc(double *ptr, int len) {};
};

class Sigmoid : public Activation
{
    public:
        virtual double calc(double *ptr, int len)
        {
            printf("sigmoid\n");
            for (int i=0; i<len; ++i, ++ptr) {
                if (*ptr >= 0) {
                    *ptr = 1 / (1 + exp(*ptr));
                }
                else {
                    *ptr = exp(*ptr) / (exp(*ptr) + 1);
                }
            }
        }

        virtual floatX* grad(double *ptr, int len)
        {
            floatX *grad = new floatX[len];
            floatX *pGrad = grad;
            for (int i=0; i<len; ++i, ++pGrad) {
                *pGrad = *(ptr+i) * (1.0 - *(ptr+i));
            }
            return grad;
        }
};

class Linear : public Activation
{
    public:
        virtual double calc(double *ptr, int len)
        {
            printf("Linear\n");
        }

        virtual floatX* grad(double *ptr, int len)
        {
        }
};

#endif
