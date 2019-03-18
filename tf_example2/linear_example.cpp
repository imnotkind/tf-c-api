#include "haebin.h"



int ModelCreate(model_t* model, const char* graph_def_filename) {
	model->status = TF_NewStatus();
	model->graph = TF_NewGraph();

	{
		// Create the session.
		TF_SessionOptions* opts = TF_NewSessionOptions();
		model->session = TF_NewSession(model->graph, opts, model->status);
		TF_DeleteSessionOptions(opts);
		if (!Okay(model->status)) return 0;
	}

	TF_Graph* g = model->graph;

	{
		// Import the graph.
		TF_Buffer* graph_def = ReadFile(graph_def_filename);
		if (graph_def == NULL) return 0;
		cout << "Read GraphDef of " << graph_def->length << " bytes" << endl;
		TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();
		TF_GraphImportGraphDef(g, graph_def, opts, model->status);
		TF_DeleteImportGraphDefOptions(opts);
		TF_DeleteBuffer(graph_def);
		if (!Okay(model->status)) return 0;
	}

	// Handles to the interesting operations in the graph.
	model->input.oper = TF_GraphOperationByName(g, "input");
	model->input.index = 0;
	model->target.oper = TF_GraphOperationByName(g, "target");
	model->target.index = 0;
	model->output.oper = TF_GraphOperationByName(g, "output");
	model->output.index = 0;

	model->init_op = TF_GraphOperationByName(g, "init");
	model->train_op = TF_GraphOperationByName(g, "train");
	model->save_op = TF_GraphOperationByName(g, "save/control_dependency");
	model->restore_op = TF_GraphOperationByName(g, "save/restore_all");

	model->checkpoint_file.oper = TF_GraphOperationByName(g, "save/Const");
	model->checkpoint_file.index = 0;
	return 1;
}

void ModelDestroy(model_t* model) {
	TF_DeleteSession(model->session, model->status);
	Okay(model->status);
	TF_DeleteGraph(model->graph);
	TF_DeleteStatus(model->status);
}

int ModelInit(model_t* model) {
	const TF_Operation* init_op[1] = { model->init_op };
	TF_SessionRun(model->session, NULL,
		/* No inputs */
		NULL, NULL, 0,
		/* No outputs */
		NULL, NULL, 0,
		/* Just the init operation */
		init_op, 1,
		/* No metadata */
		NULL, model->status);
	return Okay(model->status);
}
int ModelCheckpoint(model_t* model, const char* checkpoint_prefix, int type) {
	TF_Tensor* t = ScalarStringTensor(checkpoint_prefix, model->status);
	if (!Okay(model->status)) {
		TF_DeleteTensor(t);
		return 0;
	}
	TF_Output inputs[1] = { model->checkpoint_file };
	TF_Tensor* input_values[1] = { t };
	const TF_Operation* op[1] = { type == SAVE ? model->save_op
											  : model->restore_op };
	TF_SessionRun(model->session, NULL, inputs, input_values, 1,
		/* No outputs */
		NULL, NULL, 0,
		/* The operation */
		op, 1, NULL, model->status);
	TF_DeleteTensor(t);
	return Okay(model->status);
}

int ModelPredict(model_t* model, float* batch, int batch_size) {
	// batch consists of 1x1 matrices.
	const int64_t dims[3] = { batch_size, 1, 1 };
	const size_t nbytes = batch_size * sizeof(float);
	TF_Tensor* t = TF_AllocateTensor(TF_FLOAT, dims, 3, nbytes);
	memcpy(TF_TensorData(t), batch, nbytes);

	TF_Output inputs[1] = { model->input };
	TF_Tensor* input_values[1] = { t };
	TF_Output outputs[1] = { model->output };
	TF_Tensor* output_values[1] = { NULL };

	TF_SessionRun(model->session, NULL, inputs, input_values, 1, outputs,
		output_values, 1,
		/* No target operations to run */
		NULL, 0, NULL, model->status);
	TF_DeleteTensor(t);
	if (!Okay(model->status)) return 0;

	if (TF_TensorByteSize(output_values[0]) != nbytes) {
		cerr << "ERROR: Expected predictions tensor to have " << nbytes << " bytes, has " << TF_TensorByteSize(output_values[0]) << endl;
		TF_DeleteTensor(output_values[0]);
		return 0;
	}
	float* predictions = (float*)malloc(nbytes);
	memcpy(predictions, TF_TensorData(output_values[0]), nbytes);
	TF_DeleteTensor(output_values[0]);

	printf("Predictions:\n");
	for (int i = 0; i < batch_size; ++i) {
		cout << "    x = " << batch[i] << ", predicted y = " << predictions[i] << endl;
	}
	free(predictions);
	return 1;
}

void NextBatchForTraining(TF_Tensor** inputs_tensor,
	TF_Tensor** targets_tensor) {
#define BATCH_SIZE 10
	float inputs[BATCH_SIZE] = { 0 };
	float targets[BATCH_SIZE] = { 0 };
	for (int i = 0; i < BATCH_SIZE; ++i) {
		inputs[i] = (float)rand() / (float)RAND_MAX;
		targets[i] = 3.0 * inputs[i] + 2.0;
	}
	const int64_t dims[] = { BATCH_SIZE, 1, 1 };
	size_t nbytes = BATCH_SIZE * sizeof(float);
	*inputs_tensor = TF_AllocateTensor(TF_FLOAT, dims, 3, nbytes);
	*targets_tensor = TF_AllocateTensor(TF_FLOAT, dims, 3, nbytes);
	memcpy(TF_TensorData(*inputs_tensor), inputs, nbytes);
	memcpy(TF_TensorData(*targets_tensor), targets, nbytes);
#undef BATCH_SIZE
}

int ModelRunTrainStep(model_t* model) {
	TF_Tensor *x, *y;
	NextBatchForTraining(&x, &y);
	TF_Output inputs[2] = { model->input, model->target };
	TF_Tensor* input_values[2] = { x, y };
	const TF_Operation* train_op[1] = { model->train_op };
	TF_SessionRun(model->session, NULL, inputs, input_values, 2,
		/* No outputs */
		NULL, NULL, 0, train_op, 1, NULL, model->status);
	TF_DeleteTensor(x);
	TF_DeleteTensor(y);
	return Okay(model->status);
}
