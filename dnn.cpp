#include <string.h>
#include <time.h>
#include <exception>
#include <stddef.h>
#include <unistd.h>
#include "dnn.h"
#define DEBUG false

#define LAST_LAYER (numLayer - 1)

#define GET_MACRO(_0,_1,_2,_3,_4,_5,_6,NAME,...) NAME
#define GET(...) GET_MACRO(__VA_ARGS__, GET3D, null, GET2D)(__VA_ARGS__)

#define GET2D(mat, d1, d2, x, y) (*(mat + (x)*(d2) + (y)))
#define GET3D(mat, d1, d2, d3, x, y, z) (*(mat + (x)*(d2)*(d3) + (y)*(d3) + (z)))

#define ROOT_GATHER_ALL(...) \
    sendBuf.rank = (this->world_rank); \
    sprintf(sendBuf.msg, __VA_ARGS__); \
    MPI_Gather(&sendBuf, 1, Mpi_signal, recvBuf, 1, Mpi_signal, 0, dup_comm_world)

#define DISPLAY_SIGNAL \
    if (this->world_rank == 0) { \
        for (int i=0; i<this->world_size; ++i) { \
            printf("%s", (recvBuf[i]).msg); \
        } \
    }

#if DEBUG == true
#define DLOG(...) \
    printf(__VA_ARGS__);

#define DISP_GATHER_ALL(...) \
    ROOT_GATHER_ALL(__VA_ARGS__); \
    DISPLAY_SIGNAL
#else
#define DISP_GATHER_ALL(...)
#define DLOG(...)
#endif

#define PRINTA(_x, _len) \
    for (int i=0; i<(_len); ++i) { \
        printf("[%d]%lf ", world_rank, (_x)[i]); \
    }

DNN::DNN(int argc, char **argv)
{
    this->batchSize = 1;
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
    initNeuronSet();
    formMPIGroup();

    // Allocate weight
    this->allocWeight();
    this->allocBiases();
    this->dXidz = NULL;
    this->dXidw = NULL;
    this->dXidb = NULL;
    this->dXidtheta = NULL; // (Will be used in backprop)
    this->dzuds = NULL;
    this->z = NULL;
    this->zPrev = NULL;
    this->Y = NULL;
    this->accuracy = 0;
    this->global_loss = 0;

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

    // Duplicate communicator
    MPI_Comm_dup(MPI_COMM_WORLD, &dup_comm_world);

    // Get the rank and size in the original communicator
    MPI_Comm_rank(dup_comm_world, &this->world_rank);
    MPI_Comm_size(dup_comm_world, &this->world_size);

    if (this->world_rank == 0) {
        recvBuf = (Signal *)malloc(this->world_size*sizeof(Signal));
    }

    Signal my_signal;
    CreateSignalType(&my_signal);
}

void DNN::formMPIGroup()
{
    int recvColor = (curLayer << 16) | prevSplitId;
    int reduceColor = (curLayer << 16) | nextSplitId;

    MPI_Comm_split(dup_comm_world, recvColor, world_rank, &recvComm);
    MPI_Comm_split(dup_comm_world, reduceColor, world_rank, &reduceComm);

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

    // A communication group for exchanging information to the partitions in the next layer
    if (curLayer < LAST_LAYER) {
        bcastTag = (curLayer << 4) | nextSplitId;
        DLOG("[Layer %d] (reduce rank %d/%d) Partition %d broadcast to the remote leader of rank %d\n", curLayer, reduceRank, reduceSize, world_rank, masterId[curLayer+1] + (nextSplitId*split[curLayer+2]));

        ierr = MPI_Intercomm_create(reduceComm, 0, dup_comm_world,
                masterId[curLayer+1] + (nextSplitId*split[curLayer+2]), bcastTag, &nextBcastComm);
#if DEBUG == true
        MPI_Comm_rank(nextBcastComm, &bcastRank);
        MPI_Comm_size(nextBcastComm, &bcastSize);
        DLOG ("[nextBcastComm] curLayer=%d (rank %d/%d) worldRank %d -> slaves:%d+%d*%d\n", curLayer, bcastRank, bcastSize, world_rank, masterId[curLayer+1], nextSplitId, split[curLayer+2]);
#endif
    }

    // A communication group for exchanging information from the partition in the previous layer
    if (curLayer > 0) {
        bcastTag = ((curLayer - 1) << 4) | prevSplitId;
        DLOG("[Layer %d] (reduce rank %d/%d) Partition %d.\n", curLayer, reduceRank, reduceSize, world_rank);
        DLOG("[Layer %d] (recv rank %d/%d) Partition %d receive from the remote leader of %d\n", curLayer, recvRank, recvSize, world_rank, masterId[curLayer-1] + prevSplitId);

        ierr = MPI_Intercomm_create(recvComm, 0, dup_comm_world,
                masterId[curLayer-1] + prevSplitId, bcastTag, &prevBcastComm);
#if DEBUG == true
        MPI_Comm_rank(prevBcastComm, &bcastRank);
        MPI_Comm_size(prevBcastComm, &bcastSize);
        DLOG ("[prevBcastComm] curLayer=%d (rank %d/%d) worldRank %d -> master:%d\n", curLayer, bcastRank, bcastSize, world_rank, masterId[curLayer-1] + prevSplitId);
#endif
    }
}

void DNN::CreateSignalType(Signal *isignal) {
    Signal *signal = new Signal;
    int blocklens[2];              /*Block Lengths of data in structure*/
    MPI_Datatype old_types[2];     /*Data types of data in structure*/
    MPI_Aint indices[2];           /*Byte displacement of each piece of data*/
    MPI_Aint addr1, addr2, baseaddr;

    /*Set block lengths*/
    blocklens[0] = 1;
    blocklens[1] = SIGNAL_MAX_LEN;

    /*Set Data Types*/
    old_types[0] = MPI_INT;
    old_types[1] = MPI_CHAR;

    /*Set byte displacement for each piece of data in structure*/
    MPI_Get_address(signal, &baseaddr);
    MPI_Get_address(&signal->rank, &addr1);
    MPI_Get_address(signal->msg, &addr2);
    indices[0] = addr1 - baseaddr;
    indices[1] = addr2 - baseaddr;

    /*Create structure type in MPI so that we can transfer boundaries between nodes*/
    MPI_Type_create_struct(2,blocklens,indices,old_types,&(DNN::Mpi_signal));
    MPI_Type_commit(&(DNN::Mpi_signal));
    return;
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
            ++nLayer;
        }
        this->layerId[i] = nLayer;
    }
    this->curLayer = this->layerId[this->world_rank];

    DISP_GATHER_ALL("Partition %d: layer %d\n", this->world_rank, this->curLayer);
}

void DNN::initSplitId()
{
    this->prevSplitId = (world_rank - masterId[curLayer]) / split[curLayer+1];
    this->nextSplitId = (world_rank - masterId[curLayer]) % split[curLayer+1];
}

void DNN::initNeuronSet()
{
    this->numNeuronInSet = new int[this->numLayer+1];
    for (int l=0; l<this->numLayer+1; ++l) {
        this->numNeuronInSet[l] = this->numNeuron[l] / this->split[l];
    }

    int prevNumNeuron = this->numNeuron[curLayer];
    int nextNumNeuron = this->numNeuron[curLayer+1];
    int prevNumNeuronInSet = this->numNeuronInSet[curLayer];
    int nextNumNeuronInSet = this->numNeuronInSet[curLayer+1];
    int prevSplit = this->split[curLayer];
    int nextSplit = this->split[curLayer+1];

    if (prevSplitId == prevSplit-1 && nextSplitId == nextSplit-1) {
        prevEle = prevNumNeuron - prevNumNeuronInSet * (prevSplit-1);
        nextEle = nextNumNeuron - nextNumNeuronInSet * (nextSplit-1);
    }
    else if (prevSplitId == (prevSplit-1)) {
        prevEle = prevNumNeuron - prevNumNeuronInSet * (prevSplit-1);
        nextEle = nextNumNeuronInSet;
    }
    else if (nextSplitId == (nextSplit-1)) {
        prevEle = prevNumNeuronInSet;
        nextEle = nextNumNeuron - nextNumNeuronInSet * (nextSplit-1);
    }
    else {
        prevEle = prevNumNeuronInSet;
        nextEle = nextNumNeuronInSet;
    }
}

void DNN::allocWeight()
{
    DISP_GATHER_ALL("Partition %d: prevNumNeuronInSet = %d, nextNumNeuronInSet = %d, W = %d x %d\n", world_rank, prevNumNeuronInSet, nextNumNeuronInSet, prevEle, nextEle);

    this->weight = new floatX[prevEle * nextEle]();
}

void DNN::allocBiases()
{
    /* Only master partitions need to allocate bias */
    int nextNumNeur = this->numNeuron[curLayer+1];
    int nextNumNeuronInSet = this->numNeuronInSet[curLayer+1];
    int nextSplit = this->split[curLayer+1];

    if (prevSplitId == 0) {
        if (nextSplitId == nextSplit-1) {
            nextEle = nextNumNeur - nextNumNeuronInSet * (nextSplit-1);
        }
        else {
            nextEle = nextNumNeuronInSet;
        }
        this->biases = new floatX[nextEle]();
    }
}

void DNN::allocGradient()
{
    int m = data->getNumInst()/this->batchSize;  // Be attention on remainder
    this->dXidz = new floatX[m * this->nextEle]();
    DISP_GATHER_ALL("Partition: %d: alloc numInst: %d * nextEle: %d for dXidz\n", world_rank, m, nextEle);

    if (prevSplitId == 0) {
        this->dXidw = new floatX[this->prevEle * this->nextEle]();
        this->dXidb = new floatX[this->nextEle]();
        this->dXidtheta = new floatX[this->prevEle * this->nextEle + this->nextEle]();
    }
    else {
        this->dXidw = new floatX[this->prevEle * this->nextEle]();
        this->dXidtheta = new floatX[this->prevEle * this->nextEle]();
    }
}

void DNN::readInput(char *prefixFilename, char *datafile, INST_SZ numInst, INST_SZ numClass, FEAT_SZ numFeat, int labelInit, bool isFileExist)
{
    if (labelInit == -1) err(1, "Please set labelInit");

    /* master write file */
    if (world_rank == 0 && !isFileExist) {
        data = new LIBSVM(numInst, numClass, numFeat);
        int *featSet = data->initFeatSplit(numNeuron[0], split[0], prevSplitId);

        data->libsvm_read(datafile);
        data->export_split(numNeuron[0], split[0], prefixFilename);
        delete data;
    }

    /* Wait for the master until the file is prepared */
    MPI_Barrier(dup_comm_world);

    data = new LIBSVM(numInst, numClass, numFeat);
    int *featSet = data->initFeatSplit(numNeuron[0], split[0], prevSplitId);

    // The partitions w.r.t. first layer have to read file
    if (curLayer == 0) {
        data->read_split_feat(prevSplitId, prefixFilename);
        data->to_dense(featSet[prevSplitId], featSet[prevSplitId+1]);
    }
    else if (curLayer == numLayer-1 && prevSplitId == 0) {
        data->read_label(prevSplitId, prefixFilename, labelInit, world_rank);
    }

    this->allocGradient();  // This should be removed

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
    // Paired with DNN::initial()
{
    if (this->numNeuron) delete[] this->numNeuron;
    if (this->split) delete[] this->split;
    if (this->layerId) delete[] this->layerId;
    if (this->masterId) delete[] this->masterId;
    //if (this->numPartition) delete[] this->numPartition;
    if (this->numNeuronInSet) delete[] this->numNeuronInSet;
    if (this->weight) delete[] this->weight;
    //if (this->biases) delete[] this->biases;
    if (this->dXidz) delete[] this->dXidz;
    if (this->dXidw) delete[] this->dXidw;
    if (this->dXidb) delete[] this->dXidb;
    if (this->dXidtheta) delete[] this->dXidtheta;
    if (this->z) delete[] this->z;
    if (this->zPrev) delete[] this->zPrev;
    //if (this->data) delete[] this->data;
    MPI_Comm_free(&recvComm);
    MPI_Comm_free(&reduceComm);
    if (curLayer > 0) MPI_Comm_free(&prevBcastComm);
    if (curLayer < LAST_LAYER) MPI_Comm_free(&nextBcastComm);

    MPI_Finalize();
}

double DNN::feedforward(bool isTrain, bool isComputeAccuracy)
{
    int batchSize = this->batchSize;  // Function pipeline
    this->reg_coeff = data->getNumInst()/batchSize;  // Be attention on remainder
    double alpha = 1.0;
    double beta = 0.0;
    int m = data->getNumInst()/batchSize;  // Be attention on remainder
    int n = nextEle;
    int k = prevEle;
    int mn = m * n;
    int mk = m * k;
    double *s = new double[mn];
    this->z = new double[mn]; this->zPrev = new double[mk];
    // memset(this->z, 0, mn * sizeof(double));
    // memset(this->zPrev, 0, mn * sizeof(double));
    global_loss = 0;
    double regularization = 0;
    double global_regularization = 0;

    DISP_GATHER_ALL("[feedforward] Partition %d: m = %d, k(prev) = %d, n(next) = %d, mn = %d\n", world_rank, m, k, n, mn);

    if (curLayer == 0) {
        zPrev = data->getFeature();
        DLOG("[feedforward] Partition %d: data->getNumFeat() = %d\n", world_rank, data->getNumFeat());
    }
    else {
        // Received from previous layer
        MPI_Bcast(zPrev, mk, MPI_DOUBLE, 0, prevBcastComm);
        MPI_Bcast(&regularization, 1, MPI_DOUBLE, 0, prevBcastComm);
    }

    // Calculate s = X*W locally
    // X =[] m * k; W =[] k * n; s =[] m * n
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            m, n, k, alpha, zPrev, k, weight, n, beta, s, n);

    // Calculate regularization term
    regularization = cblas_ddot(k*n, weight, 1, weight, 1);

    // Calculate biases and its regularization term
    if (prevSplitId == 0) {
        cblas_daxpy(n, 1, biases, 1, s, 1);
        regularization += cblas_ddot(n, biases, 1, biases, 1);
    }

    DLOG("[feedforward] Partition %d, weight[0] = %lf, s(before allreduce)%lf\n", world_rank, weight[0], s[0]);

    // Sum up the s for each hidden unit
    MPI_Allreduce(s, z, mn, MPI_DOUBLE, MPI_SUM, reduceComm);
    // Sum up the regularization term
    MPI_Reduce(&regularization, &global_regularization, 1, MPI_DOUBLE, MPI_SUM, 0, reduceComm);

    DLOG("[feedforward] Partition %d, s(after allreduce)]%lf\n", world_rank, z[0]);
    DLOG("[feedforward] Partition %d, Z]%lf\n", world_rank, 1 / (1 + exp(-z[0])));

    this->activationFunc[curLayer]->calc(z, mn);

    DLOG("[feedforward] Partition %d ,z[0] = %lf\n", world_rank, z[0]);

    // Broadcast from master to the next layer (Ensure at least one
    // partition's buffer is empty to avoid deadlock)
    if (curLayer < LAST_LAYER && prevSplitId == 0) {
        MPI_Bcast(z, mn, MPI_DOUBLE, MPI_ROOT, nextBcastComm);
        MPI_Bcast(&global_regularization, 1, MPI_DOUBLE, MPI_ROOT, nextBcastComm);
    }
    else if (curLayer < LAST_LAYER && prevSplitId != 0) {
        MPI_Bcast(z, mn, MPI_DOUBLE, MPI_PROC_NULL, nextBcastComm);
        MPI_Bcast(&global_regularization, 1, MPI_DOUBLE, MPI_PROC_NULL, nextBcastComm);
    }
    else {
        //Calculate function value
        if (prevSplitId == 0) {
            int nextSplitLastId = this->split[curLayer+1] - 1;
            int nextNumNeuronInSet = this->numNeuronInSet[curLayer+1];
            int startLbl = nextSplitId * nextNumNeuronInSet;
            int stopLbl = (nextSplitId == nextSplitLastId) ? this->data->numClass : (nextSplitId+1) * nextNumNeuronInSet;

            DLOG("[feedforward] LAST LAYER: %d (from rank %d)\n", startLbl, world_rank);

            // Calculate the loss and gradient w.r.t. z, dXidz
            Y = this->data->label;
            this->current_loss = (this->*((DNN*)this)->DNN::loss)(Y, z, &m, &n, &startLbl, &stopLbl, isTrain);

            this->current_loss = this->current_loss / m + global_regularization / (2 * reg_coeff);

            if (isComputeAccuracy) {
                Predict_Label *predicted = new Predict_Label[m];
                Predict_Label *global_predicted = new Predict_Label[m];
                getPrediction(predicted, z, &m, &n, &startLbl);

                MPI_Reduce(predicted, global_predicted, m, MPI_DOUBLE_INT, MPI_MAXLOC, 0, recvComm);

                if (world_rank == masterId[LAST_LAYER]) {
                    this->accuracy = computeAccuracy(Y, global_predicted, &m);
                    MPI_Bcast(&accuracy, 1, MPI_DOUBLE, masterId[LAST_LAYER], dup_comm_world);
                }
            }

            // Reduce to global loss and broadcast to all partition
            MPI_Allreduce(&this->current_loss, &global_loss, 1, MPI_DOUBLE, MPI_SUM, recvComm);
            MPI_Bcast(&global_loss, 1, MPI_DOUBLE, masterId[LAST_LAYER], dup_comm_world);

            // Collect dXidz for gradient evaluation(Gradient is local, so don't need to gatherv)
            if (isTrain) MPI_Bcast(dXidz, mn, MPI_DOUBLE, 0, reduceComm);

            DLOG("[feedforward] current_loss: %lf (from rank %d)\n", current_loss, world_rank);
            DLOG("[feedforward] global_loss: %lf (from rank %d)\n", global_loss, world_rank);
        }
        else {
            if (isComputeAccuracy) MPI_Bcast(&accuracy, 1, MPI_DOUBLE, masterId[LAST_LAYER], dup_comm_world);
            MPI_Bcast(&global_loss, 1, MPI_DOUBLE, masterId[LAST_LAYER], dup_comm_world);
            if (isTrain) MPI_Bcast(dXidz, mn, MPI_DOUBLE, 0, reduceComm);
        }
    }

    if (curLayer != LAST_LAYER) {
        if (isComputeAccuracy)
            MPI_Bcast(&accuracy, 1, MPI_DOUBLE, masterId[LAST_LAYER], dup_comm_world);
        MPI_Bcast(&global_loss, 1, MPI_DOUBLE, masterId[LAST_LAYER], dup_comm_world);
    }

    return global_loss;
}

void DNN::backprop()
    // Calculate gradient with whole instances
{
    double alpha = 1.0;
    double beta = 0.0;
    int s = world_rank;
    int batchSize = this->batchSize;  // Function pipeline
    int m = data->getNumInst()/batchSize;  // Be attention on remainder
    int n = nextEle;
    int k = prevEle;
    int mn = m * n;
    int mk = m * k;
    int kn = k * n;
    int msgLen = mn;

    double *zGrad = this->activationFunc[curLayer]->grad(z, mn);
    int u = numNeuron[numLayer];
    int mun = m * u * n;
    int muk = m * u * k;
    double *dXids = new double[mn]();
    this->dXidw = new double[kn]();
    this->dXidb = new double[n]();
    double *dXidw_i = new double[kn]();
    double *dXidzPrev = new double[mk]();
    double *global_dXidzPrev = new double[mk]();

    if (curLayer == LAST_LAYER) { // Can combine
        int mu = m * numNeuron[numLayer];
        //delete[] dXids;
        //dXids = new double[mn]();  // dXids is a diagonal matrix only in the last layer,
        // so we reduce the memory to mn of elements.

        // Activation gradient. Element-wise vector-vector multiplication
        double *pDXids = dXids;
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                *(pDXids+i*n+j) = *(dXidz+i*n+j) * *(zGrad+i*n+j);
            }
        }

        DNNOp_Comp_Grad(zPrev, m, k, dXids, m, n, dXidw, k, n, dXidb);

        // Calculate dXidzPrev = W * dXids
        // dXidzPrev =[] m * k; dXids =[] m * n; W =[] k * n;
        // Note that in the last layer, #neurons n = u.
        // The following line should changed.
        //cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
        //    m, k, n, alpha, dXids, n, weight, n, beta, dXidzPrev, k);

        DNNOp_Comp_ShallowError(weight, k, n, dXids, m, n, dXidzPrev, m, k);

        DNNOp_Allred_ShallowError(dXidzPrev, global_dXidzPrev, mk, MPI_DOUBLE, MPI_SUM, recvComm);

        // Broadcast gradient to shallower layer
        //msgLen = muk;
        msgLen = mk;
        if (nextSplitId == 0) {
            MPI_Bcast(&msgLen, 1, MPI_INT, MPI_ROOT, prevBcastComm);
            MPI_Bcast(global_dXidzPrev, msgLen, MPI_DOUBLE, MPI_ROOT, prevBcastComm);
        }
        else {
            MPI_Bcast(&msgLen, 1, MPI_INT, MPI_PROC_NULL, prevBcastComm);
            MPI_Bcast(global_dXidzPrev, msgLen, MPI_DOUBLE, MPI_PROC_NULL, prevBcastComm);
        }
    }
    else if (curLayer < LAST_LAYER && curLayer > 0) {
        //dXidz = new double[mun]();
        // Receive gradient from deeper layer
        MPI_Bcast(&msgLen, 1, MPI_INT, 0, nextBcastComm);
        MPI_Bcast(dXidz, msgLen, MPI_DOUBLE, 0, nextBcastComm);

        // Activation gradient. Element-wise vector-vector multiplication
        double *pDXids = dXids;
        double *pDXidz = dXidz;
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                *(pDXids++) = *(pDXidz++) * zGrad[i*n + j];
            }
        }

        DNNOp_Comp_Grad(zPrev, m, k, dXids, m, n, dXidw, k, n, dXidb);

        DNNOp_Comp_ShallowError(weight, k, n, dXids, m, n, dXidzPrev, m, k);

        DNNOp_Allred_ShallowError(dXidzPrev, global_dXidzPrev, mk, MPI_DOUBLE, MPI_SUM, recvComm);

        // Broadcast gradient to shallower layer
        //msgLen = muk;
        msgLen = mk;
        if (nextSplitId == 0) {
            MPI_Bcast(&msgLen, 1, MPI_INT, MPI_ROOT, prevBcastComm);
            MPI_Bcast(global_dXidzPrev, msgLen, MPI_DOUBLE, MPI_ROOT, prevBcastComm);
        }
        else {
            MPI_Bcast(&msgLen, 1, MPI_INT, MPI_PROC_NULL, prevBcastComm);
            MPI_Bcast(global_dXidzPrev, msgLen, MPI_DOUBLE, MPI_PROC_NULL, prevBcastComm);
        }
    }
    else if (curLayer == 0) {
        //dXidz = new double[mun]();

        // Receive gradient from deeper layer
        MPI_Bcast(&msgLen, 1, MPI_INT, 0, nextBcastComm);
        MPI_Bcast(dXidz, msgLen, MPI_DOUBLE, 0, nextBcastComm);

        // Activation gradient. Element-wise vector-vector multiplication
        double *pDXids = dXids;
        double *pDXidz = dXidz;
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                *(pDXids++) = *(pDXidz++) * zGrad[i*n + j];
            }
        }

        DNNOp_Comp_Grad(zPrev, m, k, dXids, m, n, dXidw, k, n, dXidb);
    }
}

void DNN::calcJacobian()
{
    int batchSize = this->batchSize;  // Function pipeline
    int l = data->getNumInst()/batchSize;  // Be attention on remainder
    int n = nextEle;
    int k = prevEle;
    int n_L = numNeuron[numLayer];
    int ln = l * n;
    int lun = l * n_L * n;
    int luk = l * n_L * k;
    int msgLen;

    double *dzudz = new double[lun]();
    //double *global_dzds = new double[lun]();  // This is M
    this->dzuds = new double[lun]();  // This is local M

	if (curLayer == LAST_LAYER) {
		int nextSplitLastId = this->split[curLayer+1] - 1;
		int nextNumNeuronInSet = this->numNeuronInSet[curLayer+1];
		int startLbl = nextSplitId * nextNumNeuronInSet;
		int stopLbl = (nextSplitId == nextSplitLastId) ? this->data->numClass : (nextSplitId+1) * nextNumNeuronInSet;
		assert(stopLbl - startLbl == n);
		for (int i=0; i<l; ++i) {
			for (int u=0; u<n_L; ++u) {
                for (int j=startLbl; j<stopLbl; ++j) {
					if (j == u) GET(dzudz, l, n_L, n, i, u, j - startLbl) = 1.0;
                }
            }
        }
    }
    else if (curLayer < LAST_LAYER) {
        // Receive dzudz from deeper layer
        msgLen = lun;
        MPI_Bcast(&msgLen, 1, MPI_INT, 0, nextBcastComm);
        MPI_Bcast(dzudz, msgLen, MPI_DOUBLE, 0, nextBcastComm);
    }

    // Calculate DZ
    double *zGrad = this->activationFunc[curLayer]->grad(z, ln);

    double *dzudzPrev = new double[luk]();

    double *global_dzudzPrev = new double[luk](); // Only master in recvComm demands this storage

    // Calculate M^m
    double *ptr_dzuds = dzuds, *ptr_dzudz = dzudz;
    for (int i=0; i<l; ++i) {
        for (int u=0; u<n_L; ++u) {  // This can be omit in the last layer.
            for (int j=0; j<n; ++j) {
                *(ptr_dzuds++) = *(ptr_dzudz++) * GET(zGrad, l, n, i, j);
            }
        }
    }

    delete[] dzudz;
    delete[] zGrad;

    // Compute and broadcast dzudzPrev to shallower layer
    if (curLayer != 0) {
        DNNOp_Comp_DzudzPrev(dzuds, l, n_L, n, weight, k, n, dzudzPrev, l, n_L, k);

        // Reduce dzudzPrev and broadcast to the shallower layer
        DNNOp_Reduce_DzudzPrev(dzudzPrev, global_dzudzPrev, luk, MPI_DOUBLE, MPI_SUM, 0, recvComm);

        msgLen = luk;
        if (nextSplitId == 0) {
            MPI_Bcast(&msgLen, 1, MPI_INT, MPI_ROOT, prevBcastComm);
            MPI_Bcast(global_dzudzPrev, msgLen, MPI_DOUBLE, MPI_ROOT, prevBcastComm);
        }
        else {
            MPI_Bcast(&msgLen, 1, MPI_INT, MPI_PROC_NULL, prevBcastComm);
            MPI_Bcast(global_dzudzPrev, msgLen, MPI_DOUBLE, MPI_PROC_NULL, prevBcastComm);
        }
    }

    //delete[] dzudzPrev;
    //delete[] global_dzudzPrev;
}

double* DNN::Gauss_Newton_vector(double *v)
{
    int l = data->getNumInst()/batchSize;  // Be attention on remainder
    int n = nextEle;
    int k = prevEle;

    double *Gv = sumJBJv(v);

    int len = (prevSplitId == 0) ? k*n + n : k*n;
    for (int i=0; i<len; ++i) {
        *(Gv+i) = *(Gv+i) / l + *(v+i) / reg_coeff;
    }

    return Gv;
}

double* DNN::sumJBJv(double *v)
{
    int l = data->getNumInst()/batchSize;  // Be attention on remainder
    int n_L = numNeuron[numLayer];

    // attach bias to zPrev
    if (prevSplitId == 0) {
        zPrev_bias = new double[l * (prevEle + 1)];
        int len = l * (prevEle + 1);
        double *pZPrev_bias = zPrev_bias;
        double *pZPrev = zPrev;
        for (int i=0; i<l; ++i) {
            for (int t=0; t<prevEle; ++t) {
                *(pZPrev_bias++) = *(pZPrev++);
            }
            *(pZPrev_bias++) = 1;
        }
    }

    double *Pbar = Jv(v);

    int len = n_L * l;
    for (int i=0; i<len; ++i) {
        *(Pbar+i) *= 2;
    }

    double *delta = JTv(Pbar);
    //DISP_GATHER_ALL("[sumJBJv] Partition %d: delta[0] = %lf\n", world_rank, delta[0]);

    if (prevSplitId == 0) delete[] zPrev_bias;
    delete[] Pbar;
    return delta;
}

double* DNN::Jv(double *v)
{
    int batchSize = this->batchSize;  // Function pipeline
    int l = data->getNumInst()/batchSize;  // Be attention on remainder
    int n = nextEle;
    int k = prevEle;
    int kn = k * n;
    int n_L = numNeuron[numLayer];

    double *Pbar = new double[n_L * l];

    if (prevSplitId == 0) {
        // Note that dzuds associates with biases are stored only in master partitions
        DNNOp_Comp_MVTZPrevT(dzuds, l, n_L, n, v, k+1, n, zPrev_bias, l, k+1, Pbar, n_L, l);
    }
    else {
        DNNOp_Comp_MVTZPrevT(dzuds, l, n_L, n, v, k, n, zPrev, l, k, Pbar, n_L, l);
    }

    return Pbar;
}

double* DNN::JTv(double *Pbar)
{
    int batchSize = this->batchSize;  // Function pipeline
    int l = data->getNumInst()/batchSize;  // Be attention on remainder
    int n = nextEle;
    int k = prevEle;
    int n_L = numNeuron[numLayer];

    double *delta = (prevSplitId == 0) ? new double[k*n + n] : new double[k*n];

    // corresponding to weights' gradient
    //DNNOp_Comp_ZTPbarTM(zPrev, k, l, Pbar, n_L, l, dzuds, l, n_L, n, delta, k, n);

    if (prevSplitId == 0) {
        DNNOp_Comp_ZTPbarTM(zPrev_bias, k+1, l, Pbar, n_L, l, dzuds, l, n_L, n, delta, k+1, n);
    }
    else {
        DNNOp_Comp_ZTPbarTM(zPrev, k, l, Pbar, n_L, l, dzuds, l, n_L, n, delta, k, n);
    }

    return delta;
}

void DNN::randomInit()
{
    double accum;
    int total = prevEle * nextEle;

    DISP_GATHER_ALL("[randomInit] Partition %d: %d x %d = %d\n", world_rank, prevEle, nextEle, total);

    floatX *pWei = weight;
    for (int i=0; i<total; ++i) {
        accum = 0;
        for (int c=0; c<12; ++c) accum += rand();
            *(pWei++) = accum / RAND_MAX - 6;
    }
}

void DNN::sparseInit()
{
    DLOG("sparseInit\n");
}

double DNN::linear(double *x, int len)
{
    DLOG("linear\n");
}

double DNN::sigmoid(double *ptr, int len)
{
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
    for (int i=0; i<len; ++i, ++ptr) {
        //*ptr = *((int*)ptr) & 0x7fffffff;
        if (*ptr <= 0) {
            *ptr = 0;
        }
    }
}

double DNN::tanh(double *x, int len)
{
}

double DNN::squareLoss(LABEL *label, double *x, int *inst, int *local_unit, int *startLbl, int *stopLbl, bool isGradientEval=false)
{
    double Xi = 0.0;
    double y = 0;
    //this->dXidz = new double[*inst * *unit]();  // Already declared in the function init()
    //this->dXidz_local = new double[*inst * numNeuronInSet[numLayer]]();
    //double *pGrad = dXidz_local + nextSplitId*numNeuronInSet[numLayer];
    double *pGrad = NULL;
    if (isGradientEval) {
        delete[] dXidz;
        this->dXidz = new double[*inst * *local_unit]();
        pGrad = dXidz;
    }

    //printf("[squareLoss]%d * %d = %d\n", nextSplitId, numNeuronInSet[numLayer], nextSplitId * numNeuronInSet[numLayer]);
    LABEL *ptrLbl = label;
    int convertedLbl = 0;
    for (int i=0; i<*inst; ++i) {
        convertedLbl = *(label++) - *startLbl;
        //for (int u=0; u<*local_unit; ++u) {
        for (int u=0; u<*local_unit; ++u, ++x) {
            if (u == convertedLbl) {
                y = *x - 1.0;
            }
            else {
                y = *x;
            }
            Xi += y*y;  // Loss function without regularization term
            if (isGradientEval) {
                *(pGrad + u) = 2*y;
            }
            //x++;
        }
        //pGrad += numNeuronInSet[numLayer];  // Move to next row
        pGrad += *local_unit;  // Move to next row
    }

    // Gradient of regularization term
    //dXidz_local[u] += *(weight + ) / this->C;

    DLOG("[squareLoss] Xi = %lf\n", Xi);

    return Xi;
}

double DNN::squareLossCalc(LABEL *label, double *x, int *inst, int *unit, int *startLbl, int *stopLbl)
{
    double Xi = 0.0;
    double y = 0;
    printf("squareLoss\n");
    LABEL *ptrLbl = label;
    int convertedLbl = 0;
    for (int i=0; i<*inst; ++i) {
        convertedLbl = *(label++) - *startLbl;
        for (int u=0; u<*unit; ++u) {
            y = *x;
            if (u == convertedLbl) {
                y = 1.0 - y;
            }
            Xi += y*y;
            x++;
        }
        /*
           if (convertedLbl >= 0 && convertedLbl < *stopLbl) {
           double y = 1.0 - *(x + convertedLbl);
           Xi += y * y;
           for (int u=0; u<*unit; ++u) {
           if (u == convertedLbl) continue;
           Xi += (*x)*(*x);
           x++;
           }
           }
           else {
           for (int u=0; u<*unit; ++u) {
           Xi += (*x)*(*x);
           x++;
           }
           }
           */
    }
    printf("Xi = %lf\n", Xi);
    return Xi;
}

void DNN::getPrediction(Predict_Label *pPredicted, double *pZ, int *m, int *n, int *startLbl)
{
    double max = 0;

    for (int i=0; i<*m; ++i) {
        max = *pZ;
        pPredicted->similarity = *pZ;
        pPredicted->index = *startLbl;
        DLOG("*pZ = %lf\n", *pZ);
        for (int j=1; j<*n; ++j) {
            ++pZ;
            if (*pZ > max) {
                max = *pZ;
                pPredicted->similarity = *pZ;
                pPredicted->index = j + *startLbl;
            }
            DLOG("*pZ = %lf\n", *pZ);
        }
        ++pZ;
        ++pPredicted;
    }
}

double DNN::computeAccuracy(int *label, Predict_Label *pL, int *m)
{
    double correct = 0;
    for (int i=0; i<*m; ++i) {
        if (*label == pL->index) {
            ++correct;
        }
        ++label;
        ++pL;
    }
    DLOG("Correct = %lf / %d; Acc = %lf\n", correct, *m, correct/ *m);
    return correct / *m;
}

void DNN::setInstBatch(int batchSize)
{
    this->batchSize = batchSize;
}

double* DNN::CG()
{
    int n = nextEle;
    int k = prevEle;
    int kn = k * n;
    int len = (prevSplitId == 0) ? kn + n : kn;
    double *p;
    double *r;
    double rTr, rnorm_inc;
    double xi = 1e-5;
    double alpha, beta;
    double *dd, *rr;
    double *Gv;
    double pGv;
    double *d = new double[len]();
    double g_norm = cblas_dnrm2(len, dXidtheta, 1);
    double *tmp = new double[len];

    p = new double[len]();  // Gradient vector for weight and biases
    r = new double[len]();  // Gradient vector for weight and biases
    cblas_daxpy(len, -1, dXidtheta, 1, p, 1);
    cblas_dcopy(len, p, 1, r, 1);

    for (int j=0; j<CG_MAX_ITER; ++j) {
        Gv = Gauss_Newton_vector(p);

        //DISP_GATHER_ALL("[CG] Partition %d: Gv[0] = %lf, p[0] = %lf\n", world_rank, Gv[0], p[0]);

        rTr = cblas_ddot(len, r, 1, r, 1);
        pGv = cblas_ddot(len, p, 1, Gv, 1);

        alpha = rTr / pGv;

        cblas_daxpy(len, alpha, p, 1, d, 1);
        cblas_daxpy(len, -alpha, Gv, 1, r, 1);

        //PRINTA(dXidw, prevEle * nextEle);

        rnorm_inc = cblas_dnrm2(len, r, 1);

        if (rnorm_inc <= xi * g_norm) {
            //printf("Partition %d iterate %d time; rnorm_inc = %lf; g_norm = %lf\n", world_rank, j, rnorm_inc, g_norm);
            break;
        }

        beta = (rnorm_inc * rnorm_inc) / rTr;
        cblas_dcopy(len, r, 1, tmp, 1);
        cblas_daxpy(len, beta, p, 1, tmp, 1);
        cblas_dcopy(len, tmp, 1, p, 1);
    }

    return d;
}

int DNN::line_search(double alpha, double *d)
{
    int n = nextEle;
    int kn = prevEle * nextEle;
    int len = (prevSplitId == 0) ? kn + n : kn;

    MPI_Barrier(dup_comm_world);
    //cblas_dcopy(len, weight, 1, current_weight, 1);

    if (prevSplitId == 0) {
        cblas_daxpy(kn, alpha, d, 1, weight, 1);
        cblas_daxpy(n, alpha, d + kn, 1, biases, 1);
    }
    else {
        cblas_daxpy(len, alpha, d, 1, weight, 1);
    }

    // Do feedforward after weight is updated intrinsicly.
    double old_loss = global_loss;
    double new_loss = feedforward(false, false);
    char isContinue = 1;
    int iter = 0;
    while (iter++ < MAX_LINE_SEARCH) {
        if (world_rank == masterId[LAST_LAYER]) {
            // Check stopping condition, then broadcast a signal to all
            // partitions whether the line search is continue

            double zeta = ETA * cblas_ddot(len, dXidtheta, 1, d, 1);

            DLOG("[line_search] old_loss = %lf, new_loss = %lf, global_loss = %lf, threshold_loss = %lf + %lf * %lf = %lf\n", old_loss, new_loss, global_loss, old_loss, alpha, zeta, old_loss + alpha * zeta);

            if (new_loss <= old_loss + alpha * zeta) {
                // Notify all partitions to stop
                isContinue = 0;
                MPI_Bcast(&isContinue, 1, MPI_CHAR, masterId[LAST_LAYER], dup_comm_world);
                break;
            }
            else {
                // Notify all partitions to continue
                isContinue = 1;
                MPI_Bcast(&isContinue, 1, MPI_CHAR, masterId[LAST_LAYER], dup_comm_world);
            }
        }
        else {
            // Wait for the signal
            DLOG("[line_search] masterId[LAST_LAYER] = %d\n", masterId[LAST_LAYER]);

            MPI_Bcast(&isContinue, 1, MPI_CHAR, masterId[LAST_LAYER], dup_comm_world);
            if (!isContinue) break;
        }

        alpha /= 2;
        if (prevSplitId == 0) {
            cblas_daxpy(kn, -alpha, d, 1, weight, 1);
            cblas_daxpy(n, -alpha, d + kn, 1, biases, 1);
        }
        else {
            cblas_daxpy(kn, -alpha, d, 1, weight, 1);
        }
        new_loss = feedforward(false, false);
    }

    DLOG("Partition %d: iter=%d; alpha=%lf\n", world_rank, iter, alpha);

    MPI_Barrier(dup_comm_world);
    return alpha;
}

void DNN::update(double alpha, double *d)
{
    int n = nextEle;
    int k = prevEle;
    int kn = k * n;
    if (d) {
        for (int t=0; t<k; ++t) {
            for (int j=0; j<n; ++j) {
                this->weight[t*n + j] += alpha * d[t*n + j];
            }
        }
        if (prevSplitId == 0) {
            for (int j=0; j<n; ++j) {
                this->biases[j] += alpha * d[kn + j];
            }
        }
    }
    else {
        for (int t=0; t<k; ++t) {
            for (int j=0; j<n; ++j) {
                this->weight[t*n + j] -= alpha * dXidtheta[t*n + j];
            }
        }
        if (prevSplitId == 0) {
            for (int j=0; j<n; ++j) {
                this->biases[j] -= alpha * dXidtheta[kn + j];
            }
        }
    }
}

void DNN::DNNOp_Comp_Grad(double *zPrev, int zPrev_m, int zPrev_k, double *dXids, int dXids_m, int dXids_n, double *dXidw, int dXidw_k, int dXidw_n, double *dXidb)
    // Calculate the outer product dLdW = zPrev * dXids^T, one row of z, dXids = one instance
    // dXidw =[] k * n; zPrev =[] m * k; dXids =[] m * n;
{
    assert(zPrev_m == dXids_m);
    double alpha = 1.0;
    double beta = 0.0;
    //int m = data->getNumInst()/batchSize;  // Be attention on remainder
    //int n = nextEle;
    int m = dXids_m;
    int n = dXidw_n;
    int k = prevEle;
    //int u = numNeuron[numLayer];
    int mn = dXids_m * dXidw_n;
    int mk = zPrev_m * zPrev_k;
    int kn = zPrev_k * dXidw_n;
    int msgLen = mn;
    double *dXidw_i = new double[kn]();

    /*
    for (int i=0; i<m; ++i) {
        // Outer product of zPrev^T * dXids
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                k, n, 1, alpha, zPrev+i*k, k, dXids+i*n, n, beta, dXidw_i, n);
        for (int i=0; i<kn; ++i)
            dXidw[i] += dXidw_i[i];
    }
    */
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
            k, n, m, alpha, zPrev, k, dXids, n, beta, dXidw, n);

    // Add weight gradient for regularization terms to form dfdw
    for (int i=0; i<kn; ++i) {
        dXidw[i] = dXidw[i] / m + weight[i] / reg_coeff;
    }
    cblas_dcopy(kn, dXidw, 1, dXidtheta, 1);

    // Calculate biases' gradient and form dfdb. Note that biases are stored only in master partitions
    if (prevSplitId == 0) {
        for (int r=0; r<m; ++r) {
            for (int c=0; c<n; ++c) {
                dXidb[c] += dXids[r*n + c];
            }
        }

        for (int c=0; c<n; ++c) {
            dXidb[c] = dXidb[c] / m + biases[c] / reg_coeff;
        }

        cblas_dcopy(n, dXidb, 1, dXidtheta + kn, 1);
    }

    // Organize the gradient of weight and biases
}

void DNN::DNNOp_Comp_ShallowError(double *weight, int weight_k, int weight_n, double *dXids, int dXids_m, int dXids_n, double *dXidzPrev, int dXidzPrev_m, int dXidzPrev_k)
    // Calculate dXidzPrev = dXids * W^T.
    // Dimensions -> dXids =[] m * n; W =[] k * n; dXidzPrev =[] m * k;
    //   where m: #instances; k: n_{m-1}; n: n_m.
{
    double alpha = 1.0;
    double beta = 0.0;
    int m = data->getNumInst()/batchSize;  // Be attention on remainder
    int n = nextEle;
    int k = prevEle;

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
            dXids_m, weight_k, dXids_n, alpha, dXids, dXids_n, weight, weight_n, beta, dXidzPrev, dXidzPrev_k);
    /* //Compute dXidzPrev = [] m * u * k, which is useful when calculate jacobian
    if (this->curLayer == LAST_LAYER) {
        // Calculate dXidzPrev = dXids * W, one row of z, dXids = one instance
        // dXids =[] m * n; W =[] k * n;  op( W ) =[] n * k; dXidzPrev =[] m * u * k;
        for (int i=0; i<m; ++i) {
            for (int p=0; p<n; ++p) {
                for (int j=0; j<k; ++j) {
                    dXidzPrev[i*(n*k) + j*n + p] = dXids[i*n + p] * weight[j*n + p];
                }
            }
        }
    }
    else {
        // Calculate dXidzPrev = dXids * W, one row of z, dXids = one instance
        // dXids =[] m * u * n; W =[] k * n;  op( W ) =[] n * k; dXidzPrev =[] m * u * k;
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                dXids_u * dXids_m, weight_k, dXids_n, alpha, dXids, dXids_n, weight, weight_n, beta, dXidzPrev, dXidzPrev_k);
    }
    */
}

void DNN::DNNOp_Comp_DzudzPrev(double *dzuds, int dzds_l, int dzds_n_L, int dzds_n, double *weight, int weight_k, int weight_n, double *dzudzPrev, int dzudzPrev_l, int dzudzPrev_n_L, int dzudzPrev_k)
{
    double alpha = 1.0;
    double beta = 0.0;

    /*
    int stride_inst = dzds_n_L * dzds_n;

    // Calculate dzudzPrev = dzuds * W^T along instance dimension, one row of z, dXids = one instance
    // dzuds =[] l * n_L * n; W =[] k * n;  op( W ) =[] n * k; dzudzPrev =[] l * n_L * k;
    for (int i=0; i<dzds_l; ++i) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                dzds_n_L, weight_k, dzds_n, alpha, dzuds + i*stride_inst, dzds_n, weight, weight_n, beta, dzudzPrev, dzudzPrev_k);
    }
    */
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
            dzds_n_L * dzds_l, weight_k, dzds_n, alpha, dzuds, dzds_n, weight, weight_n, beta, dzudzPrev, dzudzPrev_k);
}

void DNN::DNNOp_Comp_MVTZPrevT(double *M, int M_l, int M_n_L, int M_n, double *V, int V_k, int V_n, double *ZPrev, int ZPrev_l, int ZPrev_k, double *Pbar, int Pbar_n_L, int Pbar_l)
{
    double alpha = 1.0;
    double beta = 0.0;

    double *VZPrev = new double[V_n*ZPrev_l];

    // OP(V) =[] n * k; OP(ZPrev) =[] k * l; VZ =[] n * l;
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans,
            V_n, ZPrev_l, V_k, alpha, V, V_n, ZPrev, ZPrev_k, beta, VZPrev, ZPrev_l);

    int stride_inst = M_n_L * M_n;
    for (int i=0; i<M_l; ++i) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M_n_L, 1, M_n, alpha, M + i*stride_inst, M_n, VZPrev + i, ZPrev_l, beta, Pbar + i, Pbar_l);
    }
}

void DNN::DNNOp_Comp_ZTPbarTM(double *Z, int Z_k, int Z_l, double *Pbar, int Pbar_n_L, int Pbar_l, double *M, int M_l, int M_n_L, int M_n, double *delta, int delta_k, int delta_n)
{
    double alpha = 1.0;
    double beta = 0.0;

    double *PbarM = new double[Pbar_l * M_n];

    int stride_inst = M_n_L * M_n;
    for (int i=0; i<M_l; ++i) {
        // Pbar =[] n_L * l; M =[] n_L * n;
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                1, M_n, Pbar_n_L, alpha, Pbar + i, Pbar_l, M + i*stride_inst, M_n, beta, PbarM + i*M_n, M_n);
    }

    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
            Z_k, M_n, Z_l, alpha, Z, Z_k, PbarM, M_n, beta, delta, delta_n);
}

int DNN::DNNOp_Allred_ShallowError(void *dXidzPrev, void *global_dXidzPrev, int mk, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
    // Sum up the dXidzPrev for each hidden unit
{
    return MPI_Allreduce(dXidzPrev, global_dXidzPrev, mk, datatype, op, comm);
}

int DNN::DNNOp_Allred_Dzds(void *dzdb, void *global_dzds, int lun, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
    return MPI_Allreduce(dzdb, global_dzds, lun, datatype, op, comm);
}

int DNN::DNNOp_Reduce_DzudzPrev(void *dzudzPrev, void *global_dzudzPrev, int luk, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm)
{
    return MPI_Reduce(dzudzPrev, global_dzudzPrev, luk, datatype, op, root, comm);
}

