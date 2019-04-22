#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/public/session_options.h>
#include <iostream>
#include <string>

typedef std::vector<std::pair<std::string, tensorflow::Tensor>> tensor_dict;

/**
 * @brief load a previous store model
 * @details [long description]
 *
 * in Python run:
 *
 *    saver = tf.train.Saver(tf.global_variables())
 *    saver.save(sess, './exported/my_model')
 *    tf.train.write_graph(sess.graph, '.', './exported/graph.pb, as_text=False)
 *
 * this relies on a graph which has an operation called `init` responsible to
 * initialize all variables, eg.
 *
 *    sess.run(tf.global_variables_initializer())  # somewhere in the python
 * file
 *
 * @param sess active tensorflow session
 * @param graph_fn path to graph file (eg. "./exported/graph.pb")
 * @param checkpoint_fn path to checkpoint file (eg. "./exported/my_model",
 * optional)
 * @return status of reloading
 */
tensorflow::Status LoadModel_META(tensorflow::Session *sess, std::string graph_fn,
	std::string checkpoint_fn = "") {
	tensorflow::Status status;

	// Read in the protobuf graph we exported
	tensorflow::MetaGraphDef graph_def;
	status = ReadBinaryProto(tensorflow::Env::Default(), graph_fn, &graph_def);
	if (status != tensorflow::Status::OK()) return status;

	// create the graph in the current session
	status = sess->Create(graph_def.graph_def());
	if (status != tensorflow::Status::OK()) return status;

	// restore model from checkpoint, iff checkpoint is given
	if (checkpoint_fn != "") {
		const std::string restore_op_name = graph_def.saver_def().restore_op_name();
		const std::string filename_tensor_name =
			graph_def.saver_def().filename_tensor_name();

		tensorflow::Tensor filename_tensor(tensorflow::DT_STRING,
			tensorflow::TensorShape());
		filename_tensor.scalar<std::string>()() = checkpoint_fn;

		tensor_dict feed_dict = { {filename_tensor_name, filename_tensor} };
		status = sess->Run(feed_dict, {}, { restore_op_name }, nullptr);
		if (status != tensorflow::Status::OK()) return status;
	}
	else {
		// virtual Status Run(const std::vector<std::pair<string, Tensor> >& inputs,
		//                  const std::vector<string>& output_tensor_names,
		//                  const std::vector<string>& target_node_names,
		//                  std::vector<Tensor>* outputs) = 0;
		status = sess->Run({}, {}, { "init" }, nullptr);
		if (status != tensorflow::Status::OK()) return status;
	}

	return tensorflow::Status::OK();
}

tensorflow::Status LoadModel_PB(tensorflow::Session *sess, std::string graph_fn,
	std::string checkpoint_fn = "") {
	tensorflow::Status status;

	// Read in the protobuf graph we exported
	tensorflow::GraphDef graph_def;
	status = ReadBinaryProto(tensorflow::Env::Default(), graph_fn, &graph_def);
	if (status != tensorflow::Status::OK()) return status;

	// create the graph in the current session
	status = sess->Create(graph_def);
	if (status != tensorflow::Status::OK()) return status;

	// restore model from checkpoint, iff checkpoint is given
	if (checkpoint_fn != "") {
		const std::string restore_op_name = "save/restore_all";
		const std::string filename_tensor_name = "save/Const";

		tensorflow::Tensor filename_tensor(tensorflow::DT_STRING,
			tensorflow::TensorShape());
		filename_tensor.scalar<std::string>()() = checkpoint_fn;

		tensor_dict feed_dict = { {filename_tensor_name, filename_tensor} };
		status = sess->Run(feed_dict, {}, { restore_op_name }, nullptr);
		if (status != tensorflow::Status::OK()) return status;
	}
	else {
		// virtual Status Run(const std::vector<std::pair<string, Tensor> >& inputs,
		//                  const std::vector<string>& output_tensor_names,
		//                  const std::vector<string>& target_node_names,
		//                  std::vector<Tensor>* outputs) = 0;
		status = sess->Run({}, {}, { "init" }, nullptr);
		if (status != tensorflow::Status::OK()) return status;
	}

	return tensorflow::Status::OK();
}

int linear_example() {
	const std::string graph_fn = "models/linear_example.pb";
	const std::string checkpoint_fn = "./checkpoints/checkpoint";
	//const std::string checkpoint_fn = "";

	// prepare session
	tensorflow::Session *sess;
	tensorflow::SessionOptions options;
	TF_CHECK_OK(tensorflow::NewSession(options, &sess));
	TF_CHECK_OK(LoadModel_PB(sess, graph_fn, checkpoint_fn));

	// prepare inputs
	tensorflow::TensorShape data_shape({ 3, 1, 1 });
	tensorflow::Tensor data(tensorflow::DT_FLOAT, data_shape);

	// same as in python file
	auto data_ = data.flat<float>().data();
	for (int i = 0; i < 3; ++i) data_[i] = i+1;

	tensor_dict feed_dict = {
		{"input", data},
	};

	std::vector<tensorflow::Tensor> outputs;
	TF_CHECK_OK(sess->Run(feed_dict, { "output" },
		{}, &outputs));

	std::cout << "input           " << data.DebugString() << std::endl;
	std::cout << "output          " << outputs[0].DebugString() << std::endl;
	

	return 0;
}