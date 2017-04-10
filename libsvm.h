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
    //private:
    public:  // For verify used.
    FEAT *feat;
    IDX *idx;
    int *ptrInst;
    LABEL *label;
    INST_SZ numInst;
    FEAT_SZ numFeat;
    int numClass;
    int max_line_len;
    int featSplit;
    int *featSet;

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

    LIBSVM(INST_SZ numInst, INST_SZ numClass, FEAT_SZ numFeat) : numInst(numInst), numFeat(numFeat), max_line_len(MAX_LINE_LEN) {}

    LIBSVM(int numInst, int numClass, int numFeat, int featSplit) : numInst(numInst), numClass(numClass), numFeat(numFeat), featSplit(featSplit), max_line_len(MAX_LINE_LEN) {}

    ~LIBSVM();

    int libsvm_read(const char* datafile, INST_SZ start=0, INST_SZ stop=0);

    int read_split_feat(int featSet, char *prefixFilename);

    int read_label(int prevSplitId, char *prefixFilename, int labelInit, int rankfordebug);

    void export_split(int numNeuron, int featSplit, char *prefix);

    int to_dense(int featStart, int featStop);

    template <class T> void to_one_hot(T *mat, int dim);

    floatX* getFeatDenseMatrix(int rank);

    int* getLabel();

    FEAT* getFeature();
    INST_SZ getNumInst();
    int* initFeatSplit(int numNeuron, int split);
};

#endif
