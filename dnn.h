#ifndef _DNN_H_
#define _DNN_H_

#include "libsvm.h"
#include "libsvm.hpp"
extern "C" {
#include <cblas.h>
}
//#include "mkl.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>

#define NEWTON_ITER 2
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
        floatX *biases;     // Biases, which only stored in master partitions(prevSplitId=0). E.g. 300, 300, 1.
        floatX *grad;       // Gradient of the units in output layer
        floatX *dXidz;      // Gradient of the units in output layer
        floatX *dXidw;      // Gradient of the neurons' weight in local partitions
        floatX *dXidb;      // Gradient of the biases in local partitions
        double *dLdtheta;   // Organized of the gradient of the weight and biases
        double *global_dzds;       // This is M
        //floatX *dXids;      // Gradient of the units in output layer
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
        double* calcJBJv(double *v);
        double* Jv(double *v);
        double* JTv(double *v);
        void CG();
        void update();

        void DNNOp_Comp_Grad(double *zPrev, int zPrev_m, int zPrev_k, double *dXids, int dXids_m, int dXids_n, double *dLdw, int dLdw_m, int dLdw_n, double *dLdb);
        //void DNNOp_Recv_DeeperError(void *dLdz, int msgLen, MPI_Datatype datatype, int masterRank, MPI_Comm comm);
        void DNNOp_Comp_ShallowError(int layer, double *weight, int weight_k, int weight_n, double *dXids, int dXids_m, int dXids_n, double *dLdzPrev, int dLdzPrev_m, int dLdzPrev_k);
        int DNNOp_Allred_ShallowError(void *dLdzPrev, void *global_dLdzPrev, int mk, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
        int DNNOp_Allred_Dzds(void *dzdb, void *global_dzds, int lun, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
        int DNNOp_Allred_DzudzPrev(void *dzudzPrev, void *global_dzudzPrev, int luk, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
        void DNNOp_Comp_DzudzPrev(double *global_dzds, int global_dzdb_l, int global_dzdb_n_L, int global_dzdb_n, double *weight, int weight_k, int weight_n, double *dzudzPrev, int dzudzPrev_l, int dzudzPrev_n_L, int dzudzPrev_k);
        void DNNOp_Comp_MPTZT(double *M, int M_l, int M_n_L, int M_n, double *P, int P_k, int P_n, double *Z, int Z_l, int Z_k, double *delta, int delta_n_L, int delta_l);
        void DNNOp_Comp_ZTPbarTM(double *Z, int Z_k, int Z_l, double *Pbar, int Pbar_n_L, int Pbar_l, double *M, int M_l, int M_n_L, int M_n, double *delta, int delta_k, int delta_n);
        //void DNNOp_Bcast_ShallowError(void *dLdz, int msgLen, MPI_Datatype datatype, int rank, MPI_Comm comm);
        //void DNNOp_Comp_dLdz(int LAST_LAYER, double *dLdb, double *dLdW, int , int , double *zPrev, int , int , double *dXids, int m, int u, int n);
        //void DNNOp_Comp_JTJv(int LAST_LAYER, double *dLdb, double *dLdW, int , int , double *zPrev, int , int , double *dXids, int m, int u, int n);
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
                    *ptr = 1 / (1 + exp(-*ptr));
                }
                else {
                    *ptr = exp(*ptr) / (exp(*ptr) + 1);
                }
            }
            // return
        }

        virtual floatX* grad(double *ptr, int len)
        {
            floatX *zGrad = new floatX[len];
            floatX *pGrad = zGrad;
            printf("sigmoid's gradient\n");
            for (int i=0; i<len; ++i, ++pGrad) {
                *pGrad = *(ptr+i) * (1.0 - *(ptr+i));
            }
            return zGrad;
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
            double *zGrad = new double[len];
            for (int i=0; i<len; ++i) *(zGrad + i) = 1.0;
            return zGrad;
        }
};

#endif
