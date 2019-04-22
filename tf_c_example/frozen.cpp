#include "haebin.h"

int F_ModelCreate(model_t* model, const char* graph_def_filename)
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
	model->input.oper = TF_GraphOperationByName(g, "input/Placeholder"); //DT_FLOAT
	model->input.index = 0;
	model->input2.oper = TF_GraphOperationByName(g, "input/Placeholder_2"); //DT_BOOL
	model->input2.index = 0;
	model->output.oper = TF_GraphOperationByName(g, "output/ArgMax"); //output_type : DT_INT32
	model->output.index = 0;

	
	return 1;
}

int F_ModelPredict(model_t* model, tensor_t<float> i1, tensor_t<int> i2)
{
	TF_Tensor* t1 = TF_AllocateTensor(TF_FLOAT, i1.dims.data(), i1.dims.size(), i1.vals.size() * sizeof(float));
	memcpy(TF_TensorData(t1), i1.vals.data(), i1.vals.size() * sizeof(float));

	TF_Tensor* t2 = TF_AllocateTensor(TF_BOOL, i2.dims.data(), i2.dims.size(), i2.vals.size() * sizeof(int));
	memcpy(TF_TensorData(t2), i2.vals.data(), i2.vals.size() * sizeof(int));

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

	int64_t expected_bytes = 1 * 500 * 1024 * sizeof(int);
	
	
	if (TF_TensorByteSize(output_values[0]) != expected_bytes) {
		cerr << "ERROR: Expected predictions tensor to have " << expected_bytes << " bytes, has " << TF_TensorByteSize(output_values[0]) << endl;
		TF_DeleteTensor(output_values[0]);
		return 0;
	}
	

	const auto data = static_cast<int*>(TF_TensorData(output_values[0]));

	set<int> s;

	for (int i = 0; i < 1; i++)
	{
		for (int j = 0; j < 500; j++)
		{
			for (int k = 0; k < 1024; k++)
			{
				int z = static_cast<int>(data[j * 1024 + k]);
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

	cv::Mat pred(500, 1024, CV_32SC1, data);

	show_label_image(pred);



	TF_DeleteTensor(output_values[0]);


	return 1;
}
void F_ModelDestroy(model_t* model)
{
	TF_DeleteSession(model->session, model->status);
	Okay(model->status);
	TF_DeleteGraph(model->graph);
	TF_DeleteStatus(model->status);
}