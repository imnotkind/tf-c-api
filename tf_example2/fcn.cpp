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
	model->input.oper = TF_GraphOperationByName(g, "input_image");
	model->input.index = 0;
	model->input2.oper = TF_GraphOperationByName(g, "keep_probabilty");
	model->input2.index = 0;
	model->target.oper = TF_GraphOperationByName(g, "GTLabel");
	model->target.index = 0;
	model->output.oper = TF_GraphOperationByName(g, "Pred");
	model->output.index = 0;

	model->init_op = TF_GraphOperationByName(g, "init_1");
	model->train_op = TF_GraphOperationByName(g, "Adam");
	model->save_op = TF_GraphOperationByName(g, "save_1/control_dependency");
	model->restore_op = TF_GraphOperationByName(g, "save_1/restore_all");

	model->checkpoint_file.oper = TF_GraphOperationByName(g, "save_1/Const");
	model->checkpoint_file.index = 0;
}
int FCN_ModelInit(model_t* model)
{
	return 0;
}
int FCN_ModelCheckpoint(model_t* model, const char* checkpoint_prefix, int type)
{
	return 0;
}
int FCN_ModelPredict(model_t* model, tensor_t<float> i1, tensor_t<float> i2, Mat base_img)
{
	return 0;
}
void FCN_ModelDestroy(model_t* model)
{

}