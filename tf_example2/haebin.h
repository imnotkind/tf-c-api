#pragma once
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <io.h>
#include <iostream>

#include <c_api.h>
#include <cstdio>
#include <cstdlib>

using namespace std;

typedef struct model_t {
	TF_Graph* graph;
	TF_Session* session;
	TF_Status* status;

	TF_Output input, target, output;

	TF_Operation *init_op, *train_op, *save_op, *restore_op;
	TF_Output checkpoint_file;
} model_t;

int ModelCreate(model_t* model, const char* graph_def_filename);
void ModelDestroy(model_t* model);
int ModelInit(model_t* model);
int ModelPredict(model_t* model, float* batch, int batch_size);
int ModelRunTrainStep(model_t* model);
enum SaveOrRestore { SAVE, RESTORE };
int ModelCheckpoint(model_t* model, const char* checkpoint_prefix, int type);

int Okay(TF_Status* status);
TF_Buffer* ReadFile(const char* filename);
TF_Tensor* ScalarStringTensor(const char* data, TF_Status* status);
int DirectoryExists(const char* dirname);

TF_Buffer* ReadBufferFromFile(const char* file);