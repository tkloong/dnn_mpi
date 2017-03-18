#ifndef _LIBSVM_H_
#define _LIBSVM_H_

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include <assert.h>

#define ERR_READ -1
#define ERR_NEW -2
#define MAX_LINE_LEN 1024
#define MAX_LEN_FILENAME 128

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
    int numClass;
    int max_line_len;
    int featSplit;

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
    LIBSVM(int numInst, int numClass, int numFeat, int featSplit) : numInst(numInst), numClass(numClass), numFeat(numFeat), featSplit(featSplit), max_line_len(MAX_LINE_LEN) {}
    LIBSVM(const char* datafile, INST_SZ numInst, INST_SZ numClass, FEAT_SZ numFeat) : numInst(numInst), numFeat(numFeat), max_line_len(MAX_LINE_LEN)
    {
        libsvm_read_dense(datafile, numClass, numFeat);
    }

    int libsvm_read_dense(const char* datafile, INST_SZ numClass, FEAT_SZ numFeat, INST_SZ start=0, INST_SZ stop=0);

    int read_split_feat(int featSet, char *prefixFilename, int rankfordebug);

    int libsvm_read_sparse(const char* datafile, INST_SZ numInst, INST_SZ numClass, FEAT_SZ numFeat, INST_SZ start=0, INST_SZ stop=0);

    void export_split(int numNeuron, int featSplit, char *prefix);

    floatX* getFeatDenseMatrix(int rank);

    int* getLabel();
};

#endif
