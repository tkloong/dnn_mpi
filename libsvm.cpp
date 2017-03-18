#include "libsvm.h"

int LIBSVM::libsvm_read_dense(const char* datafile, INST_SZ numClass, FEAT_SZ numFeat, INST_SZ start, INST_SZ stop)
{
    char *line;
    FILE *fp = fopen(datafile, "r");
    if (fp == NULL) return ERR_READ;

    int numInst = 0;
    while ((line = readline(fp))!=NULL) {
        ++numInst;
    }

    assert(this->numInst == numInst);

    // Initialize
    //this->label = safeMalloc(LABEL, this->numInst);
    this->label = new LABEL[this->numInst];
    if (this->label == NULL) return ERR_NEW;
    this->idx = new IDX[this->numInst * numFeat];
    if (this->idx == NULL) return ERR_NEW;
    this->feat = new FEAT[this->numInst * numFeat];
    if (this->feat == NULL) return ERR_NEW;
    this->ptrInst = new int[this->numInst + 1];
    if (this->ptrInst == NULL) return ERR_NEW;

    char *temp;
    LABEL instId = 0;
    IDX idx = 0;
    fseek(fp, 0, SEEK_SET);
    while ((line = readline(fp))!=NULL) {
        temp = strtok(line, " \t");
        *(this->label + instId) = atoi(temp);
        while(true) {
            temp = strtok(NULL, ":\n");
            if (temp == NULL) break;

            *(this->idx + idx) = atoi(temp);
            *(this->feat + idx) = strtod(strtok(NULL, " \t"), NULL);

            ++idx;
        } 
        ++instId;
        *(this->ptrInst + instId) = idx;
    }
    printf("instId= %d\n", *(ptrInst+numInst));
    fclose(fp);
}

int LIBSVM::read_split_feat(int featSet, char *prefixFilename, int rankfordebug)
{
    char *line;
    char filename[MAX_LEN_FILENAME];
    snprintf(filename, sizeof(filename), "%s.feat%d", prefixFilename, featSet);
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) err(1, "Can't open file");

    this->numInst = 0;
    while ((line = readline(fp))!=NULL) {
        ++this->numInst;
    }

    // Initialize
    int numFeatSplit = numFeat / featSplit + 1;
    this->idx = new IDX[this->numInst * numFeatSplit];
    if (this->idx == NULL) err(2, "Allocation error");
    this->feat = new FEAT[this->numInst * numFeatSplit];
    if (this->feat == NULL) err(2, "Allocation error");
    this->ptrInst = new int[this->numInst + 1];
    if (this->ptrInst == NULL) err(2, "Allocation error");

    char *temp;
    int numInst = 0;
    IDX idx = 0;
    fseek(fp, 0, SEEK_SET);
    while ((line = readline(fp))!=NULL) {
        temp = strtok(line, " :\n");
        while(true) {
            if (temp == NULL) break;

            *(this->idx + idx) = atoi(temp);
            *(this->feat + idx) = strtod(strtok(NULL, " \t"), NULL);

            temp = strtok(NULL, " :\n");
            ++idx;
        } 
        ++numInst;
        *(this->ptrInst + numInst) = idx;
    }
    printf("rank %d: read_split_feat: this->numInst=%d, numInst=%d\n", rankfordebug, this->numInst, numInst);
    printf("read_split_feat: instId= %d\n", *(ptrInst+numInst));
    assert(this->numInst == numInst);

    fclose(fp);
}

int LIBSVM::libsvm_read_sparse(const char* datafile, INST_SZ numInst, INST_SZ numClass, FEAT_SZ numFeat, INST_SZ start, INST_SZ stop)
{
}

void LIBSVM::export_split(int numNeuron, int featSplit, char *prefix)
{
    int numInst = this->numInst;
    int numFeat = this->numFeat;
    int *label = this->label;
    int *ptrInst = this->ptrInst;
    char filename[MAX_LEN_FILENAME] = {0};
    floatX *feat = this->feat;
    //floatX *data = new floatX[*(ptrInst+numInst) - 1];
    int itr = 1;

    int normSet = numNeuron / featSplit;
    int lastSet = numNeuron - (numNeuron / featSplit) * (featSplit - 1);
    int *featSet = new int[featSplit+1];
    floatX *ptrFeat;
    int *ptrIdx;
    featSet[0] = 1;
    for (int s=1; s<=featSplit; ++s) featSet[s] = featSet[s-1] + normSet;
    featSet[featSplit] = featSet[featSplit-1] + lastSet;

    for (int s=0; s<featSplit; ++s) {
        ptrInst = this->ptrInst;
        snprintf(filename, sizeof(filename), "%s%s%d", prefix, ".feat", s);
        FILE *fp = fopen(filename, "w");

        for (int i=0; i<numInst; ++i) {
            itr = featSet[s]; // feature index start by 1
            ptrFeat = feat + *ptrInst;
            ptrIdx = idx + *ptrInst;
            // Check the features between current instance index and next instance index
            //for (ptrIdx=idx+*ptrInst; *ptrIdx<*(ptrInst+1); ++ptrIdx, ++ptrFeat) {
            for (int j=*ptrInst; j<*(ptrInst+1); ++j) {
                if (itr >= featSet[s+1]) break;

                if (*ptrIdx == itr) {
                    fprintf(fp, "%d:", itr);
                    fprintf(fp, "%g ", *ptrFeat);
                    ++ptrIdx;
                    ++ptrFeat;
                }
                if (*ptrIdx < itr && *ptrIdx < featSet[s]) {
                    ++ptrIdx;
                    ++ptrFeat;
                }

                if (*ptrIdx > featSet[s]) {
                    ++itr;
                }
            }
            fprintf(fp, "\n");
            ++ptrInst;
        }
        fclose(fp);
        }

        LABEL *tmpLabel = label;
        snprintf(filename, sizeof(filename), "%s%s", prefix, ".lbl");
        FILE *fp = fopen(filename, "w");
        for (int i=0; i<numInst; ++i) {
            fprintf(fp, "%d\n", *(tmpLabel++));
        }
        fclose(fp);
    }

    floatX* LIBSVM::getFeatDenseMatrix(int rank)
    {
        int numInst = this->numInst;
        int numFeat = this->numFeat;
        int *label = this->label;
        int *ptr = this->ptrInst;
        floatX *feat = this->feat;
        floatX *data = new floatX[*(ptrInst+numInst) - 1];
        //floatX *data;
        int itr = 0;

        /*
           for (int i=0; i<numInst; ++i) {
           for (int j=*ptr; j<*(ptr+1); ++j) {
           printf("%d:", *(idx++));
           printf("%f ", *(feat++));
           if (itr == *(idx++)) {
           data[itr] = *(feat++);
           }
           else {
           data[itr] = 0;
           }
           ++itr;
           }
           printf("\n");
           ++ptr;
           }
           */
        return data;
    }

    int* LIBSVM::getLabel()
    {
        return this->label;
    }
