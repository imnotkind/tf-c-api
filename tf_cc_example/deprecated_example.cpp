// Example of training the model created by main.py in a C++ program.
//
// See also
// https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/label_image/main.cc

/*
#include <iostream>
#include <vector>
#include <cstdlib>
#include <string>
#include <sys/stat.h>

#include "third_party/tensorflow/core/framework/graph.proto.h"
#include "third_party/tensorflow/core/framework/tensor.h"
#include "third_party/tensorflow/core/lib/io/path.h"
#include "third_party/tensorflow/core/platform/env.h"
#include "third_party/tensorflow/core/platform/init_main.h"
#include "third_party/tensorflow/core/platform/logging.h"
#include "third_party/tensorflow/core/platform/types.h"
#include "third_party/tensorflow/core/public/session.h"

class Model {
public:
	Model(const string& graph_def_filename) {
		tensorflow::GraphDef graph_def;
		TF_CHECK_OK(tensorflow::ReadBinaryProto(tensorflow::Env::Default(),
			graph_def_filename, &graph_def));
		session_.reset(tensorflow::NewSession(tensorflow::SessionOptions()));
		TF_CHECK_OK(session_->Create(graph_def));
	}

	void Init() { TF_CHECK_OK(session_->Run({}, {}, { "init" }, nullptr)); }

	void Restore(const string& checkpoint_prefix) {
		SaveOrRestore(checkpoint_prefix, "save/restore_all");
	}

	void Predict(const std::vector<float>& batch) {
		std::vector<tensorflow::Tensor> out_tensors;
		TF_CHECK_OK(session_->Run({ {"input", MakeTensor(batch)} }, { "output" }, {},
			&out_tensors));
		for (int i = 0; i < batch.size(); ++i) {
			std::cout << "\t x = " << batch[i]
				<< ", predicted y = " << out_tensors[0].flat<float>()(i)
				<< "\n";
		}
	}

	void RunTrainStep(const std::vector<float>& input_batch,
		const std::vector<float>& target_batch) {
		TF_CHECK_OK(session_->Run({ {"input", MakeTensor(input_batch)},
								   {"target", MakeTensor(target_batch)} },
			{}, { "train" }, nullptr));
	}

	void Checkpoint(const string& checkpoint_prefix) {
		SaveOrRestore(checkpoint_prefix, "save/control_dependency");
	}

private:
	tensorflow::Tensor MakeTensor(const std::vector<float>& batch) {
		tensorflow::Tensor t(tensorflow::DT_FLOAT,
			tensorflow::TensorShape({ (int)batch.size(), 1, 1 }));
		for (int i = 0; i < batch.size(); ++i) {
			t.flat<float>()(i) = batch[i];
		}
		return t;
	}
	void SaveOrRestore(const string& checkpoint_prefix, const string& op_name) {
		tensorflow::Tensor t(tensorflow::DT_STRING, tensorflow::TensorShape());
		t.scalar<string>()() = checkpoint_prefix;
		TF_CHECK_OK(session_->Run({ {"save/Const", t} }, {}, { op_name }, nullptr));
	}

	std::unique_ptr<tensorflow::Session> session_;
};

bool DirectoryExists(const string& dir) {
	struct stat buf;
	return stat(dir.c_str(), &buf) == 0;
}

int main(int argc, char* argv[]) {
	const string graph_def_filename =
		"/usr/local/google/home/ashankar/tmp/gist/graph.pb";
	const string checkpoint_dir = "/usr/local/google/home/ashankar/tmp/gist/checkpoints";
	const string checkpoint_prefix = checkpoint_dir + "/checkpoint";
	bool restore = DirectoryExists(checkpoint_dir);

	// Setup global state for TensorFlow.
	tensorflow::port::InitMain(argv[0], &argc, &argv);

	std::cout << "Loading graph\n";
	Model model(graph_def_filename);

	if (!restore) {
		std::cout << "Initializing model weights\n";
		model.Init();
	}
	else {
		std::cout << "Restoring model weights from checkpoint\n";
		model.Restore(checkpoint_prefix);
	}

	const std::vector<float> testdata({ 1.0, 2.0, 3.0 });
	std::cout << "Initial predictions\n";
	model.Predict(testdata);

	std::cout << "Training for a few steps\n";
	for (int i = 0; i < 200; ++i) {
		std::vector<float> train_inputs, train_targets;
		for (int j = 0; j < 10; j++) {
			train_inputs.push_back(static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX));
			train_targets.push_back(3 * train_inputs.back() + 2);
		}
		model.RunTrainStep(train_inputs, train_targets);
	}

	std::cout << "Updated predictions\n";
	model.Predict(testdata);

	std::cout << "Saving checkpoint\n";
	model.Checkpoint(checkpoint_prefix);

	return 0;
}
*/