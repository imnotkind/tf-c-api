#include "haebin.h"

int FCN_ModelCreate(model_t* model, const char* graph_def_filename)
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
	model->input.oper = TF_GraphOperationByName(g, "input_image"); //DT_FLOAT // (a,b,c,3) : RGB (not BGR)
	model->input.index = 0;
	model->input2.oper = TF_GraphOperationByName(g, "keep_probabilty"); //DT_FLOAT // scalar
	model->input2.index = 0;
	model->target.oper = TF_GraphOperationByName(g, "GTLabel"); //DT_INT32 // (a,b,c,1)
	model->target.index = 0;
	model->output.oper = TF_GraphOperationByName(g, "Pred"); //DT_INT64 // (a,b,c)
	model->output.index = 0;

	model->init_op = TF_GraphOperationByName(g, "init_1");
	model->train_op = TF_GraphOperationByName(g, "Adam");
	model->save_op = TF_GraphOperationByName(g, "save_1/control_dependency");
	model->restore_op = TF_GraphOperationByName(g, "save_1/restore_all");

	model->checkpoint_file.oper = TF_GraphOperationByName(g, "save_1/Const");
	model->checkpoint_file.index = 0;

	return 1;
}
int FCN_ModelInit(model_t* model)
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
int FCN_ModelCheckpoint(model_t* model, const char* checkpoint_prefix, int type)
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


int FCN_ModelPredict(model_t* model, tensor_t<float> i1, tensor_t<float> i2)
{
	TF_Tensor* t1 = TF_AllocateTensor(TF_FLOAT, i1.dims.data(), i1.dims.size(), i1.vals.size() * sizeof(float));
	memcpy(TF_TensorData(t1), i1.vals.data(), i1.vals.size() * sizeof(float));

	TF_Tensor* t2 = TF_AllocateTensor(TF_FLOAT, i2.dims.data(), i2.dims.size(), i2.vals.size() * sizeof(float));
	memcpy(TF_TensorData(t2), i2.vals.data(), i2.vals.size() * sizeof(float));

	TF_Output inputs[2] = { model->input, model->input2 };
	TF_Tensor* input_values[2] = { t1, t2 };
	TF_Output outputs[1] = { model->output };
	TF_Tensor* output_values[1] = { NULL };


	TF_SessionRun(model->session,
		NULL, // Run options.
		inputs, input_values, 2, // Input tensors, input tensor values, number of inputs.
		outputs, output_values, 1, // Output tensors, output tensor values, number of outputs.
		nullptr, 0, // Target operations, number of targets.
		nullptr, // Run metadata.
		model->status // Output status.
	);
	TF_DeleteTensor(t1);
	TF_DeleteTensor(t2);
	if (!Okay(model->status)) return 0;

	int64_t expected_bytes = i1.dims[0] * i1.dims[1] * i1.dims[2] * sizeof(int64_t); //output : DT_INT64  


	if (TF_TensorByteSize(output_values[0]) != expected_bytes) {
		cerr << "ERROR: Expected predictions tensor to have " << expected_bytes << " bytes, has " << TF_TensorByteSize(output_values[0]) << endl;
		TF_DeleteTensor(output_values[0]);
		return 0;
	}


	const auto data = static_cast<int64_t*>(TF_TensorData(output_values[0]));
	auto data2 = new int[i1.dims[0] * i1.dims[1] * i1.dims[2]];
	set<int> s;

	for (int i = 0; i < i1.dims[0]; i++)
	{
		for (int j = 0; j < i1.dims[1]; j++)
		{
			for (int k = 0; k < i1.dims[2]; k++)
			{
				int z = static_cast<int>(data[i * i1.dims[1] * i1.dims[2] + j * i1.dims[2] + k]);
				data2[i * i1.dims[1] * i1.dims[2] + j * i1.dims[2] + k] = z;
				s.insert(z);
			}
		}
	}

	cout << "pred_labels : [ ";
	for (auto p : s)
	{
		cout << p << " ";
	}
	cout << "]" << endl;

	cv::Mat pred(i1.dims[1], i1.dims[2], CV_32SC1, data2);
	

	show_label_image(pred);

	delete[] data2;



	TF_DeleteTensor(output_values[0]);

	return 1;
}



int FCN_ModelRunTrainStep(model_t* model, tensor_t<float> i1, tensor_t<float> i2, tensor_t<int> i3) {

	TF_Tensor* t1 = TF_AllocateTensor(TF_FLOAT, i1.dims.data(), i1.dims.size(), i1.vals.size() * sizeof(float));
	memcpy(TF_TensorData(t1), i1.vals.data(), i1.vals.size() * sizeof(float));

	TF_Tensor* t2 = TF_AllocateTensor(TF_FLOAT, i2.dims.data(), i2.dims.size(), i2.vals.size() * sizeof(float));
	memcpy(TF_TensorData(t2), i2.vals.data(), i2.vals.size() * sizeof(float));

	TF_Tensor* t3 = TF_AllocateTensor(TF_INT32, i3.dims.data(), i3.dims.size(), i3.vals.size() * sizeof(int));
	memcpy(TF_TensorData(t3), i3.vals.data(), i3.vals.size() * sizeof(int));



	
	TF_Output inputs[3] = { model->input, model->input2, model->target };
	TF_Tensor* input_values[3] = { t1, t2, t3 };
	const TF_Operation* train_op[1] = { model->train_op };


	TF_SessionRun(model->session,
		NULL, // Run options.
		inputs, input_values, 3, // Input tensors, input tensor values, number of inputs.
		NULL, NULL, 0, // Output tensors, output tensor values, number of outputs.
		train_op, 1, // Target operations, number of targets.
		NULL, // Run metadata.
		model->status // Output status.
	);


	TF_DeleteTensor(t1);
	TF_DeleteTensor(t2);
	TF_DeleteTensor(t3);
	return Okay(model->status);
}

void FCN_ModelDestroy(model_t* model)
{
	TF_DeleteSession(model->session, model->status);
	Okay(model->status);
	TF_DeleteGraph(model->graph);
	TF_DeleteStatus(model->status);
}