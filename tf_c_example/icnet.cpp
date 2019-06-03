#include "haebin.h"

int ICNET_ModelCreate(model_t* model, const char* graph_def_filename)
{
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
	model->input.oper = TF_GraphOperationByName(g, "Placeholder"); //DT_FLOAT // (a,b,c,3) : BGR
	model->input.index = 0;
	model->output.oper = TF_GraphOperationByName(g, "Reshape_1"); //DT_INT32 // (a,b,c,3)
	model->output.index = 0;

	model->init_op = TF_GraphOperationByName(g, "init");
	model->save_op = TF_GraphOperationByName(g, "save/control_dependency");
	model->restore_op = TF_GraphOperationByName(g, "save/restore_all");


	model->checkpoint_file.oper = TF_GraphOperationByName(g, "save/Const");
	model->checkpoint_file.index = 0;

	return 1;
}
int ICNET_ModelInit(model_t* model)
{
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
int ICNET_ModelCheckpoint(model_t* model, const char* checkpoint_prefix, int type)
{
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


int ICNET_ModelPredict(model_t* model, tensor_t<float> i1)
{
	TF_Tensor* t1 = TF_AllocateTensor(TF_FLOAT, i1.dims.data(), i1.dims.size(), i1.vals.size() * sizeof(float));
	memcpy(TF_TensorData(t1), i1.vals.data(), i1.vals.size() * sizeof(float));


	TF_Output inputs[1] = { model->input };
	TF_Tensor* input_values[1] = { t1 };
	TF_Output outputs[1] = { model->output };
	TF_Tensor* output_values[1] = { NULL };


	TF_SessionRun(model->session,
		NULL, // Run options.
		inputs, input_values, 1, // Input tensors, input tensor values, number of inputs.
		outputs, output_values, 1, // Output tensors, output tensor values, number of outputs.
		nullptr, 0, // Target operations, number of targets.
		nullptr, // Run metadata.
		model->status // Output status.
	);
	TF_DeleteTensor(t1);
	if (!Okay(model->status)) return 0;

	int expected_bytes = i1.dims[0] * i1.dims[1] * i1.dims[2] * sizeof(float); //output : DT_FLOAT32


	if (TF_TensorByteSize(output_values[0]) != expected_bytes) {
		cerr << "ERROR: Expected predictions tensor to have " << expected_bytes << " bytes, has " << TF_TensorByteSize(output_values[0]) << endl;
		TF_DeleteTensor(output_values[0]);
		return 0;
	}

	const auto data = static_cast<float*>(TF_TensorData(output_values[0]));

	cout << i1.dims[0] << i1.dims[1] << i1.dims[2] << endl;
	Mat pred_mat(i1.dims[0], i1.dims[1], CV_32FC3, data);

	pred_mat.convertTo(pred_mat, CV_8UC3);

	showimage_fromMat(pred_mat);


	TF_DeleteTensor(output_values[0]);

	return Okay(model->status);
}


void ICNET_ModelDestroy(model_t* model)
{
	TF_DeleteSession(model->session, model->status);
	Okay(model->status);
	TF_DeleteGraph(model->graph);
	TF_DeleteStatus(model->status);
}