#include <string.h>
#include <time.h>
#include <exception>
#include "dnn.h"
#include <stddef.h>

#define ROOT_GATHER_ALL(...) \
    sendBuf.rank = (this->world_rank); \
    sprintf(sendBuf.msg, __VA_ARGS__); \
    MPI_Gather(&sendBuf, 1, Mpi_signal, recvBuf, 1, Mpi_signal, 0, MPI_COMM_WORLD)

#define DISPLAY_SIGNAL \
    if (this->world_rank == 0) { \
        for (int i=0; i<this->world_size; ++i) { \
            printf("%s", (recvBuf[i]).msg); \
        } \
    }

#define DISP_GATHER_ALL(...) \
    ROOT_GATHER_ALL(__VA_ARGS__); \
    DISPLAY_SIGNAL

DNN::DNN()
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

    if (this->world_rank == 0) {
        recvBuf = (Signal *)malloc(this->world_size*sizeof(Signal));
    }

    Signal my_signal;
    CreateSignalType(&my_signal);
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
    //MPI_Aint array_of_displacements[] = { offsetof( Signal, rank ),
    //                                        offsetof( Signal, msg ) };
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
            //this->lastIdInGrp[nLayer] = i;
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
    /*
       prevEle = prevNumNeurInSet;
       nextEle = nextNumNeurInSet;
       */
    DISP_GATHER_ALL("Partition %d: prevEle = %d, nextEle = %d, W = %d x %d\n", world_rank, prevNumNeurInSet, nextNumNeurInSet, prevEle, nextEle);
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
        this->biases = new floatX[nextEle]();
    }
}

void DNN::allocGradient()
{
    int m = data->getNumInst()/this->batchSize;  // Be attention on remainder
    DISP_GATHER_ALL("Partition: %d: numInst: %d, nextEle: %d\n", world_rank, m, nextEle);
    this->dXidZ = new floatX[m * this->nextEle]();
    this->dXidW = new floatX[this->prevEle * this->nextEle]();
    this->dXidb = new floatX[m * this->nextEle]();
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
    MPI_Barrier(MPI_COMM_WORLD);

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

    this->allocGradient();

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
    if (this->numNeuron)
        delete[] this->numNeuron;
    if (this->split)
        delete[] this->split;
    //if (this->z)
    //    delete[] this->z;
    MPI_Comm_free(&recvComm);
    MPI_Comm_free(&reduceComm);
    //MPI_Comm_free(&prevBcastComm);
    //MPI_Comm_free(&nextBcastComm);
    MPI_Finalize();
}

void DNN::feedforward(bool isTrain)
{
    int batchSize = this->batchSize;  // Function pipeline
    double alpha = 1.0;
    double beta = 0.0;
    int m = data->getNumInst()/batchSize;  // Be attention on remainder
    int n = nextEle;
    int k = prevEle;
    int mn = m * n;
    int mk = m * k;
    int msgLen = mn;
    double *s = new double[mn];
    this->z = new double[mn];
    this->zPrev = new double[mk];

    DISP_GATHER_ALL("[feedforward] Partition %d: m = %d, k(prev) = %d, n(next) = %d, mn = %d\n", world_rank, m, k, n, mn);

    if (curLayer == 0) {
        // Pipelined
        //for (int b=0; b<batchSize; ++b) {
        X = data->getFeature();
        printf("%d: numFeat = %d\n", world_rank, data->getNumFeat());

        // Calculate s = X*W locally, one row of X = one instance
        // X: m rows by k cols
        // W: k rows by n cols
        // s: m rows by n cols
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
        MPI_Bcast(zPrev, msgLen, MPI_DOUBLE, 0, prevBcastComm);

        // Calculate s = X*W locally, one row of X = one instance
        // zPrev: m rows by k cols
        // W: k rows by n cols
        // s: m rows by n cols
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, alpha, zPrev, k, weight, n, beta, s, n);

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
        MPI_Bcast(zPrev, msgLen, MPI_DOUBLE, 0, prevBcastComm);

        printf("msgLen in last layer = %d, mk=%d\n", msgLen, m*k);
        //Calculate s = X*W locally, one row of X = one instance
        // zPrev: m rows by k cols
        // W: k rows by n cols
        // s: m rows by n cols
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, alpha, zPrev, k, weight, n, beta, s, n);

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
    int msgLen = 1;
    int s = world_rank;

    // Calculate first

    if (curLayer == numLayer - 1) { // Can combine 
        // Sum up the s for each hidden unit
        //MPI_Allreduce(s, global, mn, MPI_DOUBLE, MPI_SUM, reduceComm);

        // Broadcast gradient to shallower layer
        if (nextSplitId == 0) {
            MPI_Bcast(&msgLen, 1, MPI_INT, MPI_ROOT, prevBcastComm);
            MPI_Bcast(&s, msgLen, MPI_INT, MPI_ROOT, prevBcastComm);
            printf("Rank %d sent %d to..\n", world_rank, s);
        }
        else {
            MPI_Bcast(&msgLen, 1, MPI_INT, MPI_PROC_NULL, prevBcastComm);
            MPI_Bcast(&s, msgLen, MPI_INT, MPI_PROC_NULL, prevBcastComm);
        }
    }
    else if (curLayer < numLayer - 1 && curLayer > 0) {
        // Receive gradient from deeper layer
        // waitPrevLayer(s, nextBcastComm);
        MPI_Bcast(&msgLen, 1, MPI_INT, 0, nextBcastComm);
        MPI_Bcast(&s, msgLen, MPI_INT, 0, nextBcastComm);
        printf("Rank %d recv %d from ..\n", world_rank, s);
        s = world_rank;

        // Complete calculation with the gradient

        // Sum up the s for each hidden unit
        //MPI_Allreduce(s, global, mn, MPI_DOUBLE, MPI_SUM, reduceComm);

        // Broadcast gradient to shallower layer
        if (nextSplitId == 0) {
            MPI_Bcast(&msgLen, 1, MPI_INT, MPI_ROOT, prevBcastComm);
            MPI_Bcast(&s, msgLen, MPI_INT, MPI_ROOT, prevBcastComm);
            printf("Rank %d sent %d to..\n", world_rank, s);
        }
        else {
            MPI_Bcast(&msgLen, 1, MPI_INT, MPI_PROC_NULL, prevBcastComm);
            MPI_Bcast(&s, msgLen, MPI_INT, MPI_PROC_NULL, prevBcastComm);
        }
    }
    else if (curLayer == 0) {
        // Receive gradient from deeper layer
        // waitPrevLayer(s, nextBcastComm);
        MPI_Bcast(&msgLen, 1, MPI_INT, 0, nextBcastComm);
        MPI_Bcast(&s, msgLen, MPI_INT, 0, nextBcastComm);
        printf("Rank %d recv %d from ..\n", world_rank, s);
        s = world_rank;

        // Complete calculation with the gradient

        // Sum up the s for each hidden unit
        //MPI_Allreduce(s, global, mn, MPI_DOUBLE, MPI_SUM, reduceComm);
    }
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
    printf("sparseInit\n");
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
    this->dXidZ = new double[*inst * *unit]();
    double *pGrad = this->dXidZ;

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
            *(pGrad++) = 2*y;
            x++;
        }
    }

    // Gradient of regularization term
    //dXidZ[u] += *(weight + ) / this->C;

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
    this->batchSize = batchSize;
}

