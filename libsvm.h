#ifndef _LIBSVM_H_
#define _LIBSVM_H_

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define ERR_READ -1
#define ERR_NEW -2
#define MAX_LINE_LEN 1024

//typedef double FEAT;
typedef int IDX;
typedef int LABEL;
typedef int INST_SZ;
typedef int FEAT_SZ;
typedef double floatX;
typedef floatX FEAT;

class LIBSVM {
    public:
    FEAT *feat;
    IDX *idx;
    int *ptrInst;
    LABEL *label;
    INST_SZ numInst;
    FEAT_SZ numFeat;
    int max_line_len;

    private:
    char* readline(FILE *input)
    {
        int len; 
        char *line = new char[max_line_len];

        if(fgets(line,max_line_len,input) == NULL)
            return NULL;

        while(strrchr(line,'\n') == NULL)
        {    
            max_line_len *= 2;
            line = (char *) realloc(line,max_line_len);
            len = (int) strlen(line);
            if(fgets(line+len,max_line_len-len,input) == NULL)
                break;
        }    
        return line;
    }

    public:
    LIBSVM() : max_line_len(MAX_LINE_LEN) {}
    LIBSVM(const char* datafile, INST_SZ numInst, INST_SZ numLabel, FEAT_SZ numFeat) : numInst(0), numFeat(numFeat), max_line_len(MAX_LINE_LEN)
    {
        libsvm_read_dense(datafile, numInst, numLabel, numFeat);
    }

    int libsvm_read_dense(const char* datafile, INST_SZ numInst, INST_SZ numLabel, FEAT_SZ numFeat, INST_SZ start=0, INST_SZ stop=0)
    {
        char *line;
        FILE *fp = fopen(datafile, "r");
        if (fp == NULL) return ERR_READ;

        while ((line = readline(fp))!=NULL) {
            ++this->numInst;
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
    }

    int libsvm_read_sparse(const char* datafile, INST_SZ numInst, INST_SZ numLabel, FEAT_SZ numFeat, INST_SZ start=0, INST_SZ stop=0)
    {
    }

    void exportSplit(int numNeuron, int featSplit, char *prefix)
    {
        int numInst = this->numInst;
        int numFeat = this->numFeat;
        int *label = this->label;
        int *ptrInst = this->ptrInst;
        char filename[128] = {0};
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

    floatX* getFeatDenseMatrix(int rank)
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

    int* getLabel()
    {
        return this->label;
    }
};

#endif
