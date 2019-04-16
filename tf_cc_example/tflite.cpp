#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/string_util.h"
#include "tensorflow/contrib/lite/tools/mutable_op_resolver.h"

int main() {
	const char graph_path[14] = "xorGate.lite";
	const int num_threads = 1;
	std::string input_layer_type = "float";
	std::vector<int> sizes = { 2 };
	float x, y;

	std::unique_ptr<tflite::FlatBufferModel> model(
		tflite::FlatBufferModel::BuildFromFile(graph_path));

	if (!model) {
		printf("Failed to mmap model\n")
			exit(0);
	}

	tflite::ops::builtin::BuiltinOpResolver resolver;
	std::unique_ptr<tflite::Interpreter> interpreter;
	tflite::InterpreterBuilder(*model, resolver)(&interpreter);

	if (!interpreter) {
		printf("Failed to construct interpreter\n");
		exit(0);
	}
	interpreter->UseNNAPI(false);

	if (num_threads != 1) {
		interpreter->SetNumThreads(num_threads);
	}

	int input = interpreter->inputs()[0];
	interpreter->ResizeInputTensor(0, sizes);

	if (interpreter->AllocateTensors() != kTfLiteOk) {
		printf("Failed to allocate tensors\n");
		exit(0);
	}

	//read two numbers

	std::printf("Type two float numbers : ");
	std::scanf("%f %f", &x, &y);
	interpreter->typed_tensor<float>(0)[0] = x;
	interpreter->typed_tensor<float>(0)[1] = y;

	printf("hello\n");
	fflush(stdout);
	if (interpreter->Invoke() != kTfLiteOk) {
		std::printf("Failed to invoke!\n");
		exit(0);
	}
	float* output;
	output = interpreter->typed_output_tensor<float>(0);
	printf("output = %f\n", output[0]);
	return 0;
}