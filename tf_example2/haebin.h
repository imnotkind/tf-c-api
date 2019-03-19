#pragma once

#include <iostream>
#include <vector>
#include <set>
#include <string>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <Windows.h>

#include <c_api.h>
#include "opencv_helper.h"


using namespace std;

typedef struct model_t {
	TF_Graph* graph;
	TF_Session* session;
	TF_Status* status;

	TF_Output input, target, output;
	TF_Output input2;

	TF_Operation *init_op, *train_op, *save_op, *restore_op;
	TF_Output checkpoint_file;
} model_t;

template <typename T>
struct tensor_t {
	std::vector<std::int64_t> dims;
	std::vector<T> vals;
};

enum SaveOrRestore { SAVE, RESTORE };



//example linear model graph (linear_example.pb)
int ModelCreate(model_t* model, const char* graph_def_filename);
void ModelDestroy(model_t* model);
int ModelInit(model_t* model);
int ModelPredict(model_t* model, float* batch, int batch_size);
int ModelRunTrainStep(model_t* model);
void NextBatchForTraining(TF_Tensor** inputs_tensor,TF_Tensor** targets_tensor);
int ModelCheckpoint(model_t* model, const char* checkpoint_prefix, int type);


//Frozen inference graph + opencv image as input (mymodel.pb)
int F_ModelCreate(model_t* model, const char* graph_def_filename);
int F_ModelPredict(model_t* model, tensor_t<float> i1, tensor_t<int> i2, Mat base_img);
void F_ModelDestroy(model_t* model);


//FCN train + inference
int FCN_ModelCreate(model_t* model, const char* graph_def_filename);
int FCN_ModelInit(model_t* model);
int FCN_ModelCheckpoint(model_t* model, const char* checkpoint_prefix, int type);
int FCN_ModelPredict(model_t* model, tensor_t<float> i1, tensor_t<float> i2, Mat base_img);
void FCN_ModelDestroy(model_t* model);



inline int Okay(TF_Status* status) {
	if (TF_GetCode(status) != TF_OK) {
		cerr << "ERROR: " << TF_Message(status) << endl;
		return 0;
	}
	return 1;
}

inline TF_Buffer* ReadFile(const char* filename) {

	const auto f = std::fopen(filename, "rb");
	if (f == nullptr) {
		return nullptr;
	}

	std::fseek(f, 0, SEEK_END);
	const auto fsize = ftell(f);
	std::fseek(f, 0, SEEK_SET);

	if (fsize < 1) {
		std::fclose(f);
		return nullptr;
	}

	const auto data = malloc(fsize);
	std::fread(data, fsize, 1, f);
	std::fclose(f);

	TF_Buffer* ret = TF_NewBufferFromString(data, fsize);
	free(data);
	return ret;
}

inline TF_Tensor* ScalarStringTensor(const char* str, TF_Status* status) {
	size_t nbytes = 8 + TF_StringEncodedSize(strlen(str));
	TF_Tensor* t = TF_AllocateTensor(TF_STRING, NULL, 0, nbytes);
	void* data = TF_TensorData(t);
	memset(data, 0, 8);  // 8-byte offset of first string.
	TF_StringEncode(str, strlen(str), (char*)data + 8, nbytes - 8, status);
	return t;
}

inline int DirectoryExists(const char* dirname) {
	/* linux
	struct stat buf;
	return stat(dirname, &buf) == 0;
	*/
	DWORD dwAttrib = GetFileAttributesA(dirname);

	return (dwAttrib != INVALID_FILE_ATTRIBUTES &&
		(dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
}