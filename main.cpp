#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define ERR_READ -1
#define ERR_NEW -2
#define MAX_ITER 200
#define MAX_LINE_LEN 1024

typedef double FEAT;
typedef int IDX;
typedef int LABEL;
typedef int INST_SZ;
typedef int FEAT_SZ;

class LIBSVM {
    FEAT *feat;
    IDX *idx;
    LABEL *label;
    INST_SZ numInst;
    int max_line_len;

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
    LIBSVM(const char* datafile, INST_SZ numInst, INST_SZ numLabel, FEAT_SZ numFeat) : numInst(0), max_line_len(MAX_LINE_LEN)
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
        this->label = new LABEL[this->numInst];
        if (this->label == NULL) return ERR_NEW;
        this->idx = new IDX[this->numInst * numFeat];
        if (this->idx == NULL) return ERR_NEW;
        this->feat = new FEAT[this->numInst * numFeat];
        if (this->feat == NULL) return ERR_NEW;

        char *temp;
        LABEL instId = 0;
        IDX idx = 0;
        fseek(fp, 0, SEEK_SET);
        while ((line = readline(fp))!=NULL) {
            temp = strtok(line, " \t");
            *(this->label + instId) = atoi(temp);
            while(true) {
                temp = strtok(NULL, ":");
                if (temp == NULL) break;

                *(this->idx + idx) = atoi(temp);
                *(this->feat + idx) = strtod(strtok(NULL, " \t"), NULL);
                //p = strtok(line, " \t");
                
                ++idx;
            } 
            ++instId;
        }
    }
};

int main(int argc, char** argv) {
    int rank; // Get the rank of the process
    int size; // Get the number of processes
    int name_len;
    char processor_name[MPI_MAX_PROCESSOR_NAME]; // Get the name of the processor

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(processor_name, &name_len);

    // Read Data
    char datafile[20] = "testing";
    INST_SZ numInst = 5000;
    INST_SZ numLabel = 26;
    FEAT_SZ numFeat = 16;
    LIBSVM data(datafile, numInst, numLabel, numFeat);

    //initial(weight, biases);

    for (int i=0; i<MAX_ITER; ++i) {
        /*
        feedforward();
        backforward();
        calcJacobian();
        calcJBJv();
        CG();
        update();
        */
    }

    // Print a hello world message
    printf("Hello world from processor %s, rank %d"
           " out of %d processors\n", processor_name, rank, size);

    MPI_Finalize();
}

