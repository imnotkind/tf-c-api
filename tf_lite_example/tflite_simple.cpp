#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/string_util.h"

int simple()
{
	const char * path_to_model = "mobilefcn.lite";
	std::unique_ptr<tflite::FlatBufferModel> model(tflite::FlatBufferModel::BuildFromFile(path_to_model));
	tflite::ops::builtin::BuiltinOpResolver resolver;
	std::unique_ptr<tflite::Interpreter> interpreter;
	tflite::InterpreterBuilder(*model, resolver)(&interpreter);
	// Resize input tensors, if desired.
	interpreter->AllocateTensors();
	float* input = interpreter->typed_input_tensor<float>(0);
	// Fill `input`.
	interpreter->Invoke();
	float* output = interpreter->typed_output_tensor<float>(0);
}