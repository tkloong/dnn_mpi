#include <string.h>
#include <time.h>
#include "dnn.h"
#define DEBUG true

#define GET_MACRO(_0,_1,_2,_3,_4,_5,_6,NAME,...) NAME
#define GET(...) GET_MACRO(__VA_ARGS__, GET3D, null, GET2D)(__VA_ARGS__)

#define GET2D(mat, d1, d2, x, y) (*(mat + (x)*(d2) + (y)))
#define GET3D(mat, d1, d2, d3, x, y, z) (*(mat + (x)*(d2)*(d3) + (y)*(d3) + (z)))

DNN::DNN()
{
    this->instBatch = 1;
    weightInit = &DNN::NOT_DEF;
}

void DNN::NOT_DEF() {}

void DNN::initial(int argc, char **argv, const int numLayer, int *numNeuron, int *split)
{
    // Initialize the MPI environment
    initMPI(argc, argv);

    srand(world_rank * time(NULL));

    // Initial neural network implicit structure
    this->numLayer = numLayer;
    this->numNeuron = new int[numLayer+1];
    this->split = new int[numLayer+1];
    memcpy(this->numNeuron, numNeuron, (numLayer+1)*sizeof(int));
    memcpy(this->split, split, (numLayer+1)*sizeof(int));

    // Assign worker to corresponding layer
    initLayerId();
    initSplitId();
    initNeuronSet();  // Hmmm.. this one may not useful
    formMPIGroup();

    // Allocate weight
    prevEle = 0;
    nextEle = 0;
    this->allocWeight();
    this->allocBiases();

    // Check configuration
    if (weightInit == &DNN::NOT_DEF) {
        fputs("Weight is not defined.\n", stderr);
        exit(1);
    }
    /*
    if (activationFunc == &DNN::NOT_DEF) {
        fputs("Activation function is not define\n", stderr);
        exit(1);
    }
    if (loss == &DNN::NOT_DEF) {
        fputs("Loss is not define\n", stderr);
        exit(1);
    }
    */

    // Initial weight
    (this ->* ((DNN*)this)->DNN::weightInit)();

}

void DNN::initMPI(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    // Get the rank and size in the original communicator
    MPI_Comm_rank(MPI_COMM_WORLD, &this->world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &this->world_size);
    //int name_len;
    //char processor_name[MPI_MAX_PROCESSOR_NAME]; // Get the name of the processor
    //MPI_Get_processor_name(processor_name, &name_len);
    
    // Grouping process based on split
    // Get the group of processes in MPI_COMM_WORLD
    //MPI_Group world_group;
    //MPI_Comm_group(MPI_COMM_WORLD, &world_group);
}

void DNN::formMPIGroup()
{
    /*
    int reduceRank;
    int reduceSize;
    int bcastTag = 0;
    int color = world_rank > 6;
    MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &reduceComm);
    MPI_Comm_rank(reduceComm, &reduceRank);
    MPI_Comm_size(reduceComm, &reduceSize);
    printf("rank %d: reduceColor=%d\n", world_rank, reduceRank);
    /*/
    int recvColor = (curLayer << 16) | prevSplitId;
    int reduceColor = (curLayer << 16) | nextSplitId;
    
    MPI_Comm_split(MPI_COMM_WORLD, recvColor, world_rank, &recvComm);
    MPI_Comm_split(MPI_COMM_WORLD, reduceColor, world_rank, &reduceComm);

    int bcastTag = 0;
    int recvRank;
    int recvSize;
    int reduceRank;
    int reduceSize;
    int bcastRank;
    int bcastSize;
    int ierr;
    MPI_Comm_rank(reduceComm, &reduceRank);
    MPI_Comm_size(reduceComm, &reduceSize);
    MPI_Comm_rank(recvComm, &recvRank);
    MPI_Comm_size(recvComm, &recvSize);

    if (curLayer < numLayer - 1) {
        //printf ("[reduce %d] (rank %d/%d)worldRank %d: %d\n", curLayer, reduceRank, reduceSize, world_rank, masterId[curLayer+1] + (nextSplitId*split[curLayer+2]));
        ierr = MPI_Intercomm_create(reduceComm, 0, MPI_COMM_WORLD, 
                masterId[curLayer+1] + (nextSplitId*split[curLayer+2]), bcastTag, &nextBcastComm);
        MPI_Comm_rank(nextBcastComm, &bcastRank);
        MPI_Comm_size(nextBcastComm, &bcastSize);
        //printf ("[bcastComm] curLayer=%d (rank %d/%d) worldRank %d -> slaves:%d+%d*%d\n", curLayer, bcastRank, bcastSize, world_rank, masterId[curLayer+1], nextSplitId, split[curLayer+2]);
    }

    if (curLayer > 0) {
        //printf ("[recv %d] (rank %d/%d)worldRank %d: %d\n", curLayer, recvRank, recvSize, world_rank, masterId[curLayer-1] + prevSplitId);
        ierr = MPI_Intercomm_create(recvComm, 0, MPI_COMM_WORLD, 
                masterId[curLayer-1] + prevSplitId, bcastTag, &prevBcastComm);
        MPI_Comm_rank(prevBcastComm, &bcastRank);
        MPI_Comm_size(prevBcastComm, &bcastSize);
        //printf ("[bcastComm] curLayer=%d (rank %d/%d) worldRank %d -> master:%d\n", curLayer, bcastRank, bcastSize, world_rank, masterId[curLayer-1] + prevSplitId);
    }
}

void DNN::initLayerId()
{
    int total = 0;
    this->masterId = new int[this->numLayer+1]();  // equivalent to master Id
    for (int l=1; l<this->numLayer+1; ++l) {
        total += this->split[l-1] * this->split[l];
        this->masterId[l] = total;
    }

    this->layerId = new int[world_size];  // This can be change to scalar since each
                                     // partition know its layerId is enough
    int nLayer = 0;
    for (int i=0; i<world_size; ++i) {
        if (i >= this->masterId[nLayer+1]) {
            //this->lastIdInGrp[nLayer] = i;
            ++nLayer;
        }
        this->layerId[i] = nLayer;
    }
    this->curLayer = this->layerId[this->world_rank];
}

void DNN::initSplitId()
{
    this->prevSplitId = (world_rank - masterId[curLayer]) / split[curLayer+1];
    this->nextSplitId = (world_rank - masterId[curLayer]) % split[curLayer+1];
}

void DNN::initNeuronSet()
{
    this->numNeurInSet = new int[this->numLayer+1];
    for (int l=0; l<this->numLayer+1; ++l) {
        this->numNeurInSet[l] = this->numNeuron[l] / this->split[l];
    }
}

void DNN::allocWeight()
{
    int prevNumNeur = this->numNeuron[curLayer];
    int nextNumNeur = this->numNeuron[curLayer+1];
    int prevNumNeurInSet = this->numNeurInSet[curLayer];
    int nextNumNeurInSet = this->numNeurInSet[curLayer+1];
    int prevSplit = this->split[curLayer];
    int nextSplit = this->split[curLayer+1];
    prevEle = 0;
    nextEle = 0;
    
    if (prevSplitId == prevSplit-1 && nextSplitId == nextSplit-1) {
        prevEle = prevNumNeur - prevNumNeurInSet * (prevSplit-1);
        nextEle = nextNumNeur - nextNumNeurInSet * (nextSplit-1);
    }
    else if (prevSplitId == (prevSplit-1)) {
        prevEle = prevNumNeur - prevNumNeurInSet * (prevSplit-1);
        nextEle = nextNumNeurInSet;
    }
    else if (nextSplitId == (nextSplit-1)) {
        prevEle = prevNumNeurInSet;
        nextEle = nextNumNeur - nextNumNeurInSet * (nextSplit-1);
    }
    else {
        prevEle = prevNumNeurInSet;
        nextEle = nextNumNeurInSet;
    }
    this->weight = new floatX[prevEle * nextEle];
}

void DNN::allocBiases()
{
    /* Only master partitions need to allocate bias */
    int nextNumNeur = this->numNeuron[curLayer+1];
    int nextNumNeurInSet = this->numNeurInSet[curLayer+1];
    int nextSplit = this->split[curLayer+1];

    if (prevSplitId == 0) {
        if (nextSplitId == nextSplit-1) {
            nextEle = nextNumNeur - nextNumNeurInSet * (nextSplit-1);
        }
        else {
            nextEle = nextNumNeurInSet;
        }
        this->biases = new floatX[nextEle];
    }
}

void DNN::readInput(char *prefixFilename, char *datafile, INST_SZ numInst, INST_SZ numClass, FEAT_SZ numFeat, int labelInit,  bool isFileExist)
{
    if (labelInit == -1) err(1, "Please set labelInit");

    /* master write file */
    if (world_rank == 0 && !isFileExist) {
        data = new LIBSVM(numInst, numClass, numFeat);
        int *featSet = data->initFeatSplit(numNeuron[0], split[0]);

        data->libsvm_read(datafile);
        data->export_split(numNeuron[0], split[0], prefixFilename);
        delete data;
    }

    /* Wait for the master until the file is prepared */
    MPI_Barrier(MPI_COMM_WORLD);

    data = new LIBSVM(numInst, numClass, numFeat);
    int *featSet = data->initFeatSplit(numNeuron[0], split[0]);

    // The partitions w.r.t. first layer have to read file
    if (curLayer == 0) {
        data->read_split_feat(prevSplitId, prefixFilename);
        data->to_dense(featSet[prevSplitId], featSet[prevSplitId+1]);
    }   
    else if (curLayer == numLayer-1 && prevSplitId == 0) {
        data->read_label(prevSplitId, prefixFilename, labelInit, world_rank);
    }
}

void DNN::readWeightFromFile(/* file */ char *filename)
{
    /*
    if (world_rank == l) {
        this->biases[l] = new floatX[this->numNeuron[l+1]];
    }
    */
}

void DNN::readWeight(/* array */)
{
}

void DNN::readBiases(/* array */)
{
    /*
    this->biases = new floatX*[this->numLayer];
    if (world_rank == l) {
        this->biases = new floatX[this->numNeuron[l+1]];
    }
    */
}

void DNN::finalize()
{
    delete[] this->numNeuron;
    delete[] this->split;
    MPI_Comm_free(&recvComm);
    MPI_Comm_free(&reduceComm);
    //MPI_Comm_free(&prevBcastComm);
    //MPI_Comm_free(&nextBcastComm);
    MPI_Finalize();
}

void DNN::feedforward(bool isTrain)
{
    this->C = data->getNumInst()/instBatch;  // Be attention on remainder
    int batchSize = this->instBatch;  // Function pipeline
    double alpha = 1.0;
    double beta = 0.0;
    int m = data->getNumInst()/batchSize;  // Be attention on remainder
    int n = nextEle;
    int k = prevEle;
    int mn = m * n;
    int msgLen = mn;
    double *s = new double[mn*sizeof(double)];  // linear mapping of X*W
    z = new double[mn*sizeof(double)];  // linear mapping of X*W
    printf("mn = %d\n", mn);

    if (curLayer == 0) {
        // Pipelined
        //for (int b=0; b<batchSize; ++b) {
            X = data->getFeature();

            // Calculate s = X*W locally, one row of X = one instance
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                    m, n, k, alpha, X, k, weight, n, beta, s, n);
            
            // Sum up the s for each hidden unit
            MPI_Allreduce(s, z, mn, MPI_DOUBLE, MPI_SUM, reduceComm);

            this->activationFunc[curLayer]->calc(z, mn);

            // Broadcast from master to the next layer (Ensure at least one 
            // partition's buffer is empty to avoid deadlock)
            // broadcast(s, nextBcastComm, curLayer, prevSplitId);
            msgLen = mn;
            if (curLayer < numLayer - 1 && prevSplitId == 0) {
                MPI_Bcast(&msgLen, 1, MPI_INT, MPI_ROOT, nextBcastComm);
                MPI_Bcast(s, msgLen, MPI_DOUBLE, MPI_ROOT, nextBcastComm);
            }
            else if (curLayer < numLayer - 1 && prevSplitId != 0) {
                MPI_Bcast(&msgLen, 1, MPI_INT, MPI_PROC_NULL, nextBcastComm);
                MPI_Bcast(s, msgLen, MPI_DOUBLE, MPI_PROC_NULL, nextBcastComm);
            }
            //}
    }
    else if (curLayer < numLayer - 1){
        // Received from previous layer
        // waitPrevLayer(s, prevBcastComm);
        MPI_Bcast(&msgLen, 1, MPI_INT, 0, prevBcastComm);
        MPI_Bcast(s, msgLen, MPI_DOUBLE, 0, prevBcastComm);

        // Calculate s = X*W locally, one row of X = one instance
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                m, n, k, alpha, s, k, weight, n, beta, s, n);
        
        // Sum up the s for each hidden unit
        MPI_Allreduce(s, z, mn, MPI_DOUBLE, MPI_SUM, reduceComm);

        this->activationFunc[curLayer]->calc(z, mn);

        // broadcast(s, nextBcastComm, curLayer, prevSplitId);
        msgLen = mn;
        if (prevSplitId == 0) {
            MPI_Bcast(&msgLen, 1, MPI_INT, MPI_ROOT, nextBcastComm);
            MPI_Bcast(s, msgLen, MPI_DOUBLE, MPI_ROOT, nextBcastComm);
        }
        else if (prevSplitId != 0) {
            MPI_Bcast(&msgLen, 1, MPI_INT, MPI_PROC_NULL, nextBcastComm);
            MPI_Bcast(s, msgLen, MPI_DOUBLE, MPI_PROC_NULL, nextBcastComm);
        }
    }
    else {
        //waitPrevLayer(s, prevBcastComm);
        MPI_Bcast(&msgLen, 1, MPI_INT, 0, prevBcastComm);
        MPI_Bcast(s, msgLen, MPI_DOUBLE, 0, prevBcastComm);

        //Calculate s = X*W locally, one row of X = one instance
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                m, n, k, alpha, s, k, weight, n, beta, s, n);

        // Sum up the s for each hidden unit
        MPI_Allreduce(s, z, mn, MPI_DOUBLE, MPI_SUM, reduceComm);

        this->activationFunc[curLayer]->calc(z, mn);

        //Calculate function value
        if (isTrain) {
            if (prevSplitId == 0) {
                int nextSplitLastId = this->split[curLayer+1] - 1;
                int nextNumNeurInSet = this->numNeurInSet[curLayer+1];
                int startLbl = nextSplitId * nextNumNeurInSet;
                int stopLbl = (nextSplitId == nextSplitLastId) ? this->data->numClass : (nextSplitId+1) * nextNumNeurInSet;

                printf("LAST LAYER: %d (from rank %d)\n", startLbl, world_rank);

                (this->*((DNN*)this)->DNN::loss)(this->data->label, z, &m, &n, &startLbl, &stopLbl);
            }

        }
        else {
            //predict
        }

        // MPI_Reduce(s, global, mn, MPI_DOUBLE, MPI_SUM, , funcValComm);
    }

    /*
     * dnn.loss();
    int global = world_rank;
    //MPI_Allreduce(&world_rank, &global, 1, MPI_INT, MPI_SUM, reduceComm);
    printf("[b4 bcast] rank %d: %d\n", world_rank, global);

    msgLen = mn;
    // Broadcast from master to the next layer (Ensure at least one 
    // partition's buffer is empty to avoid deadlock)
    if (curLayer < numLayer - 1 && prevSplitId == 0) {
        printf("Rank %d send %d to ..\n", world_rank, global);
        //MPI_Bcast(&global, 1, MPI_INT, MPI_ROOT, nextBcastComm);
        MPI_Bcast(&msgLen, 1, MPI_INT, MPI_ROOT, nextBcastComm);
        MPI_Bcast(s, msgLen, MPI_DOUBLE, MPI_ROOT, nextBcastComm);
    }
    else if (curLayer < numLayer - 1 && prevSplitId != 0) {
        //MPI_Bcast(&global, 1, MPI_INT, MPI_PROC_NULL, nextBcastComm);
        MPI_Bcast(&msgLen, 1, MPI_INT, MPI_PROC_NULL, nextBcastComm);
        MPI_Bcast(s, msgLen, MPI_DOUBLE, MPI_PROC_NULL, nextBcastComm);
    }
    else {
        //MPI_Bcast(&global, 1, MPI_INT, 0, prevBcastComm);
        MPI_Bcast(&msgLen, 1, MPI_INT, 0, prevBcastComm);
        MPI_Bcast(s, msgLen, MPI_DOUBLE, 0, prevBcastComm);
        printf("Rank %d recv %d from ..\n", world_rank, global);
    }
    */

    //printf("[after bcast] rank %d: %d\n", world_rank, global);
}

void DNN::backprop()
{
    double alpha = 1.0;
    double beta = 0.0;
    int s = world_rank;
    int batchSize = this->instBatch;  // Function pipeline
    int m = data->getNumInst()/batchSize;  // Be attention on remainder
    int n = nextEle;
    int k = prevEle;
    int mn = m * n;
    int mk = m * k;
    int kn = k * n;
    int msgLen = mn;

    // temporary //
    double *zPrev = new double[mk]();
    //
    double *zGrad = this->activationFunc[curLayer]->grad(z, mn);
    double *dLds = new double[mn]();
    int u = numNeuron[numLayer];
    int mun = m * u * n;
    dLdw = new double[kn]();
    dLdb = new double[n]();
    double *dLdw_i = new double[kn]();
    double *dLdzPrev = new double[mk]();
    double *global_dLdzPrev = new double[mk]();
    if (curLayer == numLayer - 1) { // Can combine
        //dLds = new double[mn]();  // dLds is a diagonal matrix only in the last layer,
        // so we reduce the memory to mn of elements.

        // Activation gradient. Element-wise vector-vector multiplication
        for (int i=0; i<mn; ++i) {
            dLds[i] = this->dLdz[i] * zGrad[i];  // dLds is a diagonal matrix only in the last layer
        }

        DNNOp_Comp_Grad(zPrev, m, k, dLds, m, n, dLdw, k, n, dLdb);

        // Calculate dLdzPrev = W * dLds
        // dLdzPrev =[] m * k; dLds =[] m * n; W =[] k * n;
        // Note that in the last layer, #neurons n = u.
        // The following line should changed.
        //cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
        //    m, k, n, alpha, dLds, n, weight, n, beta, dLdzPrev, k);

        DNNOp_Comp_ShallowError(curLayer, weight, k, n, dLds, m, n, dLdzPrev, m, k);
        //cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
        //   m, k, n, alpha, dLds, n, weight, n, beta, dLdzPrev, k);

        DNNOp_Allred_ShallowError(dLdzPrev, global_dLdzPrev, mk, MPI_DOUBLE, MPI_SUM, reduceComm);

        //MPI_Allreduce(dLdzPrev, global_dLdzPrev, mk, MPI_DOUBLE, MPI_SUM, reduceComm);

        // Broadcast gradient to shallower layer
        msgLen = mk;
        if (nextSplitId == 0) {
            MPI_Bcast(&msgLen, 1, MPI_INT, MPI_ROOT, prevBcastComm);
            MPI_Bcast(global_dLdzPrev, msgLen, MPI_DOUBLE, MPI_ROOT, prevBcastComm);
#if DEBUG == true
            printf("Rank %d sent %d to..\n", world_rank, s);
#endif
        }
        else {
            MPI_Bcast(&msgLen, 1, MPI_INT, MPI_PROC_NULL, prevBcastComm);
            MPI_Bcast(global_dLdzPrev, msgLen, MPI_DOUBLE, MPI_PROC_NULL, prevBcastComm);
        }
    }
    else if (curLayer < numLayer - 1 && curLayer > 0) {
        dLdz = new double[mn]();
        // Receive gradient from deeper layer
        // waitPrevLayer(s, nextBcastComm);
        MPI_Bcast(&msgLen, 1, MPI_INT, 0, nextBcastComm);
        MPI_Bcast(dLdz, msgLen, MPI_DOUBLE, 0, nextBcastComm);
#if DEBUG == true
        printf("Rank %d(curLayer: %d) recv %d from ..\n", world_rank, curLayer, s);
#endif
        s = world_rank;

        // Activation gradient. Element-wise vector-vector multiplication
        for (int i=0; i<mn; ++i) {
            dLds[i] = dLdz[i] * zGrad[i];
        }

        DNNOp_Comp_Grad(zPrev, m, k, dLds, m, n, dLdw, k, n, dLdb);

        DNNOp_Comp_ShallowError(curLayer, weight, k, n, dLds, m, n, dLdzPrev, m, k);

        DNNOp_Allred_ShallowError(dLdzPrev, global_dLdzPrev, mk, MPI_DOUBLE, MPI_SUM, reduceComm);

        // Broadcast gradient to shallower layer
        msgLen = mk;
        if (nextSplitId == 0) {
            MPI_Bcast(&msgLen, 1, MPI_INT, MPI_ROOT, prevBcastComm);
            MPI_Bcast(global_dLdzPrev, msgLen, MPI_DOUBLE, MPI_ROOT, prevBcastComm);
#if DEBUG == true
            printf("Rank %d sent %d to..\n", world_rank, s);
#endif
        }
        else {
            MPI_Bcast(&msgLen, 1, MPI_INT, MPI_PROC_NULL, prevBcastComm);
            MPI_Bcast(global_dLdzPrev, msgLen, MPI_DOUBLE, MPI_PROC_NULL, prevBcastComm);
        }
    }
    else if (curLayer == 0) {
        dLdz = new double[mn]();

        // Receive gradient from deeper layer
        // waitPrevLayer(s, nextBcastComm);
        MPI_Bcast(&msgLen, 1, MPI_INT, 0, nextBcastComm);
        MPI_Bcast(dLdz, msgLen, MPI_DOUBLE, 0, nextBcastComm);
#if DEBUG == true
        printf("Rank %d recv %d from ..\n", world_rank, s);
#endif
        s = world_rank;

        // Activation gradient. Element-wise vector-vector multiplication
        for (int i=0; i<mn; ++i) {
            dLds[i] = dLdz[i] * zGrad[i];
        }

        DNNOp_Comp_Grad(zPrev, m, k, dLds, m, n, dLdw, k, n, dLdb);

        // Calculate dLdzPrev = W * dLds, one row of z, dLds = one instance
        //cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
        //    m, k, n, alpha, dLds, n, weight, n, beta, dLdzPrev, k);

        // Sum up the dLdzPrev for each hidden unit (Last layer don't need to do reduce)
        //MPI_Allreduce(dLdzPrev, global_dLdzPrev, mk, MPI_DOUBLE, MPI_SUM, reduceComm);
    }
}

void DNN::calcJacobian()
{
    int batchSize = this->instBatch;  // Function pipeline
    int l = data->getNumInst()/batchSize;  // Be attention on remainder
    int n = nextEle;
    int k = prevEle;
    int n_L = numNeuron[numLayer];
    int ln = l * n;
    int lun = l * n_L * n;
    int luk = l * n_L * k;
    int msgLen = lun;

    double *dzudz = new double[lun]();
    double *dzdb = new double[lun]();  // This is local M
    global_dzds = new double[lun]();  // This is M

    if (curLayer == numLayer - 1) { // Can combine
        for (int i=0; i<l; ++i) {
            for (int j=0; j<n_L; ++j) {
                GET(dzudz, l, n_L, n_L, i, j, j) = 1.0;
            }
        }
    }
    else if (curLayer < numLayer - 1) {
        // Receive dzudz from deeper layer
        MPI_Bcast(&msgLen, 1, MPI_INT, 0, nextBcastComm);
        MPI_Bcast(dzudz, msgLen, MPI_DOUBLE, 0, nextBcastComm);
    }

    // Calculate DZ
    double *zGrad = this->activationFunc[curLayer]->grad(z, ln);
    double *dzudzPrev = new double[luk]();
    double *global_dzudzPrev = new double[luk]();

    // Calculate M^m
    double *ptr_dzdb = dzdb;
    double *ptr_dzudz = dzudz;
    for (int i=0; i<l; ++i) {
        for (int u=0; u<n_L; ++u) {
            for (int j=0; j<n; ++j) {
                *(ptr_dzdb++) = *(ptr_dzudz++) * GET(zGrad, l, n, i, j);
            }
        }
    }

    // All reduce dzdb
    DNNOp_Allred_Dzds(dzdb, global_dzds, lun, MPI_DOUBLE, MPI_SUM, reduceComm);

    // Broadcast dzudzPrev to shallower layer
    if (curLayer != 0) {
        DNNOp_Comp_DzudzPrev(global_dzds, l, n_L, n, weight, k, n, dzudzPrev, l, n_L, k);

        // All reduce dzudzPrev
        DNNOp_Allred_DzudzPrev(dzudzPrev, global_dzudzPrev, luk, MPI_DOUBLE, MPI_SUM, reduceComm);

        // Broadcast gradient to shallower layer
        msgLen = lun;
        if (nextSplitId == 0) {
            MPI_Bcast(&msgLen, 1, MPI_INT, MPI_ROOT, prevBcastComm);
            MPI_Bcast(global_dzudzPrev, msgLen, MPI_DOUBLE, MPI_ROOT, prevBcastComm);
#if DEBUG == true
#endif
        }
        else {
            MPI_Bcast(&msgLen, 1, MPI_INT, MPI_PROC_NULL, prevBcastComm);
            MPI_Bcast(global_dzudzPrev, msgLen, MPI_DOUBLE, MPI_PROC_NULL, prevBcastComm);
        }
    }

    //delete[] dzudz;
    //delete[] dzdb;
    //delete[] zGrad;
    //delete[] dzudzPrev;
    //delete[] global_dzudzPrev;
}

double* DNN::calcJBJv(double *v)
{
    double *Pbar = Jv(v);
    return JTv(Pbar);
}

double* DNN::Jv(double *v)
{
    int batchSize = this->instBatch;  // Function pipeline
    int l = data->getNumInst()/batchSize;  // Be attention on remainder
    int n = nextEle;
    int k = prevEle;
    int n_L = numNeuron[numLayer];

    double *Pbar = new double[n_L * l];

    // corresponding to weights' gradient
    DNNOp_Comp_MPTZT(global_dzds, l, n_L, n, dLdw, k, n, z, l, k, Pbar, n_L, l);

    // corresponding to biases' gradient. Note that biases are stored only in master partitions

    return Pbar;
}

double* DNN::JTv(double *Pbar)
{
    int batchSize = this->instBatch;  // Function pipeline
    int l = data->getNumInst()/batchSize;  // Be attention on remainder
    int n = nextEle;
    int k = prevEle;
    int n_L = numNeuron[numLayer];

    double *delta = new double[k * n];

    // corresponding to weights' gradient
    DNNOp_Comp_ZTPbarTM(z, k, l, Pbar, n_L, l, global_dzds, l, n_L, n, delta, k, n);

    // corresponding to biases' gradient. Note that biases are stored only in master partitions

    return delta;
}

void DNN::randomInit()
{
    double accum;
    int total = prevEle * nextEle;
#if DEBUG == true
    printf("%d: %d x %d = %d\n", world_rank, prevEle, nextEle, total);
#endif
    floatX *pWei = weight;
    for (int i=0; i<total; ++i) {
        accum = 0;
        for (int c=0; c<12; ++c) accum += rand();
        *(pWei++) = accum / RAND_MAX - 6;
    }
}

void DNN::sparseInit()
{
#if DEBUG == true
    printf("sparseInit\n");
#endif
}

double DNN::linear(double *x, int len)
{
    printf("linear\n");
}

double DNN::sigmoid(double *ptr, int len)
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

double DNN::relu(double *ptr, int len)
{
    printf("relu\n");
    for (int i=0; i<len; ++i, ++ptr) {
        //*ptr = *((int*)ptr) & 0x7fffffff;
        if (*ptr <= 0) {
            *ptr = 0;
        }
    }
}

double DNN::tanh(double *x, int len)
{
    printf("tanh\n");
}

double DNN::squareLoss(LABEL *label, double *x, int *inst, int *unit, int *startLbl, int *stopLbl)
{
    double loss = 0.0;
    double y = 0;
    this->grad = new double[*unit]();
    this->dLdz = new double[*inst * *unit]();

    printf("squareLoss\n");
    LABEL *ptrLbl = label;
    this->numNeuron[this->numLayer];
    int convertedLbl = 0;
    for (int i=0; i<*inst; ++i) {
        convertedLbl = *(label++) - *startLbl;
        for (int u=0; u<*unit; ++u) {
            y = *x;
            if (u == convertedLbl) {
                y = y - 1.0;
            }
            loss += y*y;
            dLdz[i * *unit + u] = 2*y;
            grad[u] += dLdz[i * *unit + u];
            x++;
        }
    }

    // Average gradient
    for (int u=0; u<*unit; ++u) {
        grad[u] /= this->instBatch;
        // Gradient of regularization term
        //grad[u] += *(weight + ) / this->C;
    }

    printf("loss = %lf\n", loss);
}

double DNN::squareLossCalc(LABEL *label, double *x, int *inst, int *unit, int *startLbl, int *stopLbl)
{
    double loss = 0.0;
    double y = 0;
    printf("squareLoss\n");
    LABEL *ptrLbl = label;
    this->numNeuron[this->numLayer];
    int convertedLbl = 0;
    for (int i=0; i<*inst; ++i) {
        convertedLbl = *(label++) - *startLbl;
        for (int u=0; u<*unit; ++u) {
            y = *x;
            if (u == convertedLbl) {
                y = 1.0 - y;
            }
            loss += y*y;
            x++;
        }
        /*
           if (convertedLbl >= 0 && convertedLbl < *stopLbl) {
           double y = 1.0 - *(x + convertedLbl);
           loss += y * y;
           for (int u=0; u<*unit; ++u) {
           if (u == convertedLbl) continue;
           loss += (*x)*(*x);
           x++;
           }
           }
           else {
           for (int u=0; u<*unit; ++u) {
           loss += (*x)*(*x);
           x++;
           }
           }
           */
    }
    printf("loss = %lf\n", loss);
    return loss;
}

void DNN::setInstBatch(int batchSize)
{
    this->instBatch = batchSize;
}

void DNN::CG()
{
    double *p = dLdtheta;
    double *Mv;
    for (int j=0; j<NEWTON_ITER; ++j) {
        Mv = calcJBJv(p);
    }
}

void DNN::DNNOp_Comp_Grad(double *zPrev, int zPrev_m, int zPrev_k, double *dLds, int dLds_m, int dLds_n, double *dLdw, int dLdw_m, int dLdw_n, double *dLdb)
    // Calculate the outer product dLdW = zPrev * dLds^T, one row of z, dLds = one instance
    // dLdw =[] k * n; zPrev =[] m * k; dLds =[] m * n;
{
    double alpha = 1.0;
    double beta = 0.0;
    //int m = data->getNumInst()/batchSize;  // Be attention on remainder
    //int n = nextEle;
    int m = dLdw_m;
    int n = dLdw_n;
    int k = prevEle;
    //int u = numNeuron[numLayer];
    int mn = dLdw_m * dLdw_n;
    int mk = zPrev_m * zPrev_k;
    int kn = zPrev_k * dLdw_n;
    int msgLen = mn;
    double *dLdw_i = new double[kn]();

    for (int i=0; i<m; ++i) {
        // Outer product of zPrev^T * dLds
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                k, n, 1, alpha, zPrev+i*k, k, dLds+i*n, n, beta, dLdw_i, n);
        for (int i=0; i<kn; ++i)
            dLdw[i] += dLdw_i[i];
    }

    // Add weight gradient of regularization terms
    for (int i=0; i<kn; ++i) {
        dLdw[i] += dLdw[i] / m + weight[i] / C;
    }

    // Calculate bias' gradient. Note that biases are stored only in master partitions
    if (prevSplitId == 0) {
        for (int r=0; r<m; ++r) {
            for (int c=0; c<n; ++c) {
                dLdb[c] += dLds[c];
            }
        }

        for (int c=0; c<n; ++c) {
            dLdb[c] /= m;
            dLdb[c] += biases[c] / C;
        }
    }

    // Organize the gradient of weight and biases
}

void DNN::DNNOp_Comp_ShallowError(int layer, double *weight, int weight_k, int weight_n, double *dLds, int dLds_m, int dLds_n, double *dLdzPrev, int dLdzPrev_m, int dLdzPrev_k)
    // Calculate dLdzPrev = dLds * W^T.
    // Dimensions -> dLds =[] m * n; W =[] k * n; dLdzPrev =[] m * k;
    //   where m: #instances; k: n_{m-1}; n: n_m.
{
    double alpha = 1.0;
    double beta = 0.0;

    // Calculate dLdzPrev = dLds * W, one row of z, dLds = one instance
    // dLds =[] m * n; W =[] k * n;  op( W ) =[] n * k; dLdzPrev =[] m * k;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
            dLds_m, weight_k, dLds_n, alpha, dLds, dLds_n, weight, weight_n, beta, dLdzPrev, dLdzPrev_k);
}

void DNN::DNNOp_Comp_DzudzPrev(double *global_dzds, int global_dzdb_l, int global_dzdb_n_L, int global_dzdb_n, double *weight, int weight_k, int weight_n, double *dzudzPrev, int dzudzPrev_l, int dzudzPrev_n_L, int dzudzPrev_k)
{
    double alpha = 1.0;
    double beta = 0.0;

    int stride_inst = global_dzdb_n_L * global_dzdb_n;

    // Calculate dzudzPrev = dzdb * W^T along instance dimension, one row of z, dLds = one instance
    // dzdb =[] l * n_L * n; W =[] k * n;  op( W ) =[] n * k; dzudzPrev =[] l * n_L * k;
    for (int i=0; i<global_dzdb_l; ++i) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                global_dzdb_n_L, weight_k, global_dzdb_n, alpha, global_dzds + i*stride_inst, global_dzdb_n, weight, weight_n, beta, dzudzPrev, dzudzPrev_k);
    }
}

void DNN::DNNOp_Comp_MPTZT(double *M, int M_l, int M_n_L, int M_n, double *P, int P_k, int P_n, double *Z, int Z_l, int Z_k, double *Pbar, int Pbar_n_L, int Pbar_l)
{
    double alpha = 1.0;
    double beta = 0.0;

    double *tmp = new double[P_k*Z_l];

    // OP(P) =[] n * k; OP(Z) =[] k * l; tmp =[] n * l;
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans,
            P_n, Z_l, P_k, alpha, P, P_n, Z, Z_k, beta, tmp, Z_l);

    int stride_inst = M_n_L * M_n;
    for (int i=0; i<M_l; ++i) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M_n_L, 1, M_n, alpha, M + i*stride_inst, M_n, tmp + i, Z_l, beta, Pbar + i, Pbar_l);
    }
}

void DNN::DNNOp_Comp_ZTPbarTM(double *Z, int Z_k, int Z_l, double *Pbar, int Pbar_n_L, int Pbar_l, double *M, int M_l, int M_n_L, int M_n, double *delta, int delta_k, int delta_n)
{
    double alpha = 1.0;
    double beta = 0.0;

    double *tmp = new double[Pbar_l * M_n];

    int stride_inst = M_n_L * M_n;
    for (int i=0; i<M_l; ++i) {
        // Pbar =[] n_L * l; M =[] n_L * n;
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                Pbar_n_L, M_n, 1, alpha, Pbar + i, Pbar_l, M + i*stride_inst, M_n, beta, tmp + i*M_n, M_n);
    }
}

int DNN::DNNOp_Allred_ShallowError(void *dLdzPrev, void *global_dLdzPrev, int mk, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
    // Sum up the dLdzPrev for each hidden unit
{
    return MPI_Allreduce(dLdzPrev, global_dLdzPrev, mk, datatype, op, comm);
}

int DNN::DNNOp_Allred_Dzds(void *dzdb, void *global_dzds, int lun, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
    return MPI_Allreduce(dzdb, global_dzds, lun, datatype, op, comm);
}

int DNN::DNNOp_Allred_DzudzPrev(void *dzudzPrev, void *global_dzudzPrev, int luk, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
    return MPI_Allreduce(dzudzPrev, global_dzudzPrev, luk, datatype, op, comm);
}

