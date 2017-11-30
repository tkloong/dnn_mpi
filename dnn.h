#ifndef _DNN_H_
#define _DNN_H_

#include "libsvm.h"
#include "libsvm.hpp"
//extern "C" {
//#include <cblas.h>
//}
//#include "mkl.h"
#include <mkl_cblas.h>
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <exception>

#define CG_MAX_ITER 100
#define MAX_LINE_SEARCH 30
#define SIGNAL_MAX_LEN 128

#define ETA 0.01

using namespace std;

class Activation;
class Sigmoid;
class Linear;
class DNN;
typedef void (DNN::*fpWeightInit)();
typedef double (DNN::*fpLoss)(LABEL *label, double *x, int *inst, int *unit, int *startLbl, int *stopLbl, bool isGradientEval);
typedef double floatX;
typedef struct {
    int rank;
    char msg[SIGNAL_MAX_LEN];
} Signal;
typedef struct {
    double similarity;
    int   index;
} Predict_Label;

class DNN {
    private:
        int numLayer;       // Total number of layer for this NN, excluding input layer.
        int curLayer;       // Layer Id in this partition.
        int *numNeuron;     // Array of number of neuron for this NN structure. E.g. 28-300-300-10.
        int *split;         // Split structure for this NN. E.g. 2-2-1-1.
        int *layerId;       // Array of layer Id for each partition.
        int *masterId;      // Array of master Id for each layer.
        int prevSplitId;    // Previous split id in this partition.
        int nextSplitId;    // Next split id in this partition.
        int prevEle;        // First dimension of weight matrix.
        int nextEle;        // Second dimension of weight matrix.
        int *numNeuronInSet;  // Array of floor(number of neurons in partitions for each layer). E.g. 14, 150, 300, 10
        int m;              // The cardinality of subsampled set.
        int l;              // The cardinality of instance set.
        floatX *weight;     // Weight matrix in this partition. E.g. 28*300, 300*300, 300*1.
        floatX *biases;     // Biases, which only stored in master partitions(prevSplitId=0). E.g. 300, 300, 10.
        floatX *theta;      // Organized of the weight (and bias if exists)
        floatX *dXidz;      // Gradient of the neurons in local partition w.r.t. z
        floatX *dXidw;      // Gradient of the neurons' weight in local partitions
        floatX *dXidb;      // Gradient of the biases in local partitions
        double *dXidtheta;   // Organized of the gradient of the weight and biases
        double *dzuds;       // This is M
        double *z;			// instance rows by n_m columns
        double *zPrev;		// instance rows by n_{m-1} columns
        double *zPrev_bias;		// instance rows by (n_{m-1} + 1) columns
        double current_loss;
        double global_loss;
        double accuracy;
        floatX *X;          // Array of input feature
        int *Y;             // Array of one-hot label for multiclass
        double subsampled_portion;
        int batchSize;      // For pipeline in function value evaluation
        double reg_coeff;   // Regularization coefficient, C
        int world_size;     // Number of the processes
        MPI_Comm dup_comm_world;
        MPI_Comm recvComm;  // recv from previous layer together with split[n] partitions
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
        Predict_Label *global_predicted;
        double getLoss() { return global_loss; }
        void getPrediction(Predict_Label *pPredicted, double *z, int *m, int *n, int *startLbl);
        double computeAccuracy(int *label, Predict_Label *pL, int *m);
        double getAccuracy() { return this->accuracy; }
        DNN(int argc, char **argv);
        int world_rank;     // Rank of the process
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
        double softmax(LABEL *label, double *x, int *inst, int *unit, int *startLbl, int *stopLbl);
        double logLoss(LABEL *label, double *x, int *inst, int *unit, int *startLbl, int *stopLbl);
        double squareLoss(LABEL *label, double *x, int *inst, int *unit, int *startLbl, int *stopLbl, bool isGradientEval);
        double squareLossCalc(LABEL *label, double *x, int *inst, int *unit, int *startLbl, int *stopLbl);
        double l1Loss(LABEL *label, double *x, int *inst, int *unit, int *startLbl, int *stopLbl);
        //double (*activationFunc[])(double *);
        void setInstBatch(int batchSize);
        void setSubsampledPortion(double subsampled_portion);
        void activate();
        double feedforward(bool isTrain, bool isComputeAccuracy=true);
        void backprop();
        void calcJacobian();
        double* sumJBJv(double *v);
        double* Gauss_Newton_vector(double *v);
        double* Jv(double *v);
        double* JTv(double *v);
        double* CG();
        int line_search(double alpha, double *d);
        void update(double alpha, double *d);

        void DNNOp_Comp_Grad(double *zPrev, int zPrev_m, int zPrev_k, double *dXids, int dXids_m, int dXids_n, double *dLdtheta, int dLdw_m, int dLdw_n);
        void DNNOp_Comp_ShallowError(double *weight, int weight_k, int weight_n, double *dXids, int dXids_m, int dXids_n, double *dXidzPrev, int dXidzPrev_m, int dXidzPrev_k);
        int DNNOp_Allred_ShallowError(void *dLdzPrev, void *global_dLdzPrev, int mk, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
        int DNNOp_Allred_Dzds(void *dzdb, void *global_dzds, int lun, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
        int DNNOp_Reduce_DzudzPrev(void *dzudzPrev, void *global_dzudzPrev, int luk, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm);
        void DNNOp_Comp_DzudzPrev(double *dzuds, int global_dzdb_l, int global_dzdb_n_L, int global_dzdb_n, double *weight, int weight_k, int weight_n, double *dzudzPrev, int dzudzPrev_l, int dzudzPrev_n_L, int dzudzPrev_k);
        void DNNOp_Comp_MVTZPrevT(double *M, int M_l, int M_n_L, int M_n, double *V, int V_k, int V_n, double *ZPrev, int ZPrev_l, int ZPrev_k, double *delta, int delta_n_L, int delta_l);
        void DNNOp_Comp_ZTPbarTM(double *Z, int Z_k, int Z_l, double *Pbar, int Pbar_n_L, int Pbar_l, double *M, int M_l, int M_n_L, int M_n, double *delta, int delta_k, int delta_n);
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
        }

        virtual floatX* grad(double *ptr, int len)
        {
            double *zGrad = new double[len];
            for (int i=0; i<len; ++i) *(zGrad + i) = 1.0;
            return zGrad;
        }
};

#endif
