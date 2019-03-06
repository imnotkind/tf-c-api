#include "tf_utils.hpp"
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/cvdef.h>
#include <string>
#include <set>
#include <windows.h>

using namespace std;

using namespace cv;

string type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}



void showimage_fromMat(Mat image)
{
	cout << type2str(image.type()) << endl;
	cout << image.size() << endl;
	if (image.empty())                      // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return;
	}
	namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Display window", image);                // Show our image inside it.
	waitKey(0); // Wait for a keystroke in the window
}

int practicemodel()
{
	std::cout << "TensorFlow Version: " << TF_Version() << std::endl;

	TF_Graph* graph = tf_utils::LoadGraphDef("graph.pb");
	if (graph == nullptr) {
		std::cout << "Can't load graph" << std::endl;
		return 1;
	}

	TF_Output input_op = { TF_GraphOperationByName(graph, "input_4"), 0 };
	if (input_op.oper == nullptr) {
		std::cout << "Can't init input_op" << std::endl;
		return 2;
	}

	const std::vector<std::int64_t> input_dims = { 1, 5, 12 };
	const std::vector<float> input_vals = {
	  -0.4809832f, -0.3770838f, 0.1743573f, 0.7720509f, -0.4064746f, 0.0116595f, 0.0051413f, 0.9135732f, 0.7197526f, -0.0400658f, 0.1180671f, -0.6829428f,
	  -0.4810135f, -0.3772099f, 0.1745346f, 0.7719303f, -0.4066443f, 0.0114614f, 0.0051195f, 0.9135003f, 0.7196983f, -0.0400035f, 0.1178188f, -0.6830465f,
	  -0.4809143f, -0.3773398f, 0.1746384f, 0.7719052f, -0.4067171f, 0.0111654f, 0.0054433f, 0.9134697f, 0.7192584f, -0.0399981f, 0.1177435f, -0.6835230f,
	  -0.4808300f, -0.3774327f, 0.1748246f, 0.7718700f, -0.4070232f, 0.0109549f, 0.0059128f, 0.9133330f, 0.7188759f, -0.0398740f, 0.1181437f, -0.6838635f,
	  -0.4807833f, -0.3775733f, 0.1748378f, 0.7718275f, -0.4073670f, 0.0107582f, 0.0062978f, 0.9131795f, 0.7187147f, -0.0394935f, 0.1184392f, -0.6840039f,
	};


	TF_Tensor* input_tensor = tf_utils::CreateTensor(TF_FLOAT,
		input_dims.data(), input_dims.size(),
		input_vals.data(), input_vals.size() * sizeof(float));

	TF_Output out_op = { TF_GraphOperationByName(graph, "output_node0"), 0 };
	if (out_op.oper == nullptr) {
		std::cout << "Can't init out_op" << std::endl;
		return 3;
	}

	TF_Tensor* output_tensor = nullptr;

	TF_Status* status = TF_NewStatus();
	TF_SessionOptions* options = TF_NewSessionOptions();
	TF_Session* sess = TF_NewSession(graph, options, status);
	TF_DeleteSessionOptions(options);

	if (TF_GetCode(status) != TF_OK) {
		TF_DeleteStatus(status);
		return 4;
	}

	TF_SessionRun(sess,
		nullptr, // Run options.
		&input_op, &input_tensor, 1, // Input tensors, input tensor values, number of inputs.
		&out_op, &output_tensor, 1, // Output tensors, output tensor values, number of outputs.
		nullptr, 0, // Target operations, number of targets.
		nullptr, // Run metadata.
		status // Output status.
	);

	if (TF_GetCode(status) != TF_OK) {
		std::cout << "Error run session";
		TF_DeleteStatus(status);
		return 5;
	}

	TF_CloseSession(sess, status);
	if (TF_GetCode(status) != TF_OK) {
		std::cout << "Error close session";
		TF_DeleteStatus(status);
		return 6;
	}

	TF_DeleteSession(sess, status);
	if (TF_GetCode(status) != TF_OK) {
		std::cout << "Error delete session";
		TF_DeleteStatus(status);
		return 7;
	}

	const auto data = static_cast<float*>(TF_TensorData(output_tensor));

	std::cout << "Output vals: " << data[0] << ", " << data[1] << ", " << data[2] << ", " << data[3] << std::endl;

	TF_DeleteTensor(input_tensor);
	TF_DeleteTensor(output_tensor);
	TF_DeleteGraph(graph);
	TF_DeleteStatus(status);

	return 0;
}

int practicemodel2()
{
	TF_Graph* graph = tf_utils::LoadGraphDef("graph.pb");
	if (graph == nullptr) {
		std::cout << "Can't load graph" << std::endl;
		return 1;
	}

	std::vector<TF_Output> input_op;
	input_op.push_back( { TF_GraphOperationByName(graph, "input_4"), 0 });
	if (input_op[0].oper == nullptr) {
		std::cout << "Can't init input_op" << std::endl;
		return 2;
	}

	const std::vector<std::int64_t> input_dims = { 1, 5, 12 };
	const std::vector<float> input_vals = {
	  -0.4809832f, -0.3770838f, 0.1743573f, 0.7720509f, -0.4064746f, 0.0116595f, 0.0051413f, 0.9135732f, 0.7197526f, -0.0400658f, 0.1180671f, -0.6829428f,
	  -0.4810135f, -0.3772099f, 0.1745346f, 0.7719303f, -0.4066443f, 0.0114614f, 0.0051195f, 0.9135003f, 0.7196983f, -0.0400035f, 0.1178188f, -0.6830465f,
	  -0.4809143f, -0.3773398f, 0.1746384f, 0.7719052f, -0.4067171f, 0.0111654f, 0.0054433f, 0.9134697f, 0.7192584f, -0.0399981f, 0.1177435f, -0.6835230f,
	  -0.4808300f, -0.3774327f, 0.1748246f, 0.7718700f, -0.4070232f, 0.0109549f, 0.0059128f, 0.9133330f, 0.7188759f, -0.0398740f, 0.1181437f, -0.6838635f,
	  -0.4807833f, -0.3775733f, 0.1748378f, 0.7718275f, -0.4073670f, 0.0107582f, 0.0062978f, 0.9131795f, 0.7187147f, -0.0394935f, 0.1184392f, -0.6840039f,
	};

	std::vector<TF_Tensor*> input_tensor;
	input_tensor.push_back(tf_utils::CreateTensor(TF_FLOAT,
		input_dims.data(), input_dims.size(),
		input_vals.data(), input_vals.size() * sizeof(float)));

	TF_Output out_op = { TF_GraphOperationByName(graph, "output_node0"), 0 };
	if (out_op.oper == nullptr) {
		std::cout << "Can't init out_op" << std::endl;
		return 3;
	}

	TF_Tensor* output_tensor = nullptr;

	TF_Status* status = TF_NewStatus();
	TF_SessionOptions* options = TF_NewSessionOptions();
	TF_Session* sess = TF_NewSession(graph, options, status);
	TF_DeleteSessionOptions(options);

	if (TF_GetCode(status) != TF_OK) {
		TF_DeleteStatus(status);
		return 4;
	}


	TF_SessionRun(sess,
		nullptr, // Run options.
		&input_op[0], &input_tensor[0], 1, // Input tensors, input tensor values, number of inputs.
		&out_op, &output_tensor, 1, // Output tensors, output tensor values, number of outputs.
		nullptr, 0, // Target operations, number of targets.
		nullptr, // Run metadata.
		status // Output status.
	);

	if (TF_GetCode(status) != TF_OK) {
		std::cout << "Error run session";
		TF_DeleteStatus(status);
		return 5;
	}

	TF_CloseSession(sess, status);
	if (TF_GetCode(status) != TF_OK) {
		std::cout << "Error close session";
		TF_DeleteStatus(status);
		return 6;
	}

	TF_DeleteSession(sess, status);
	if (TF_GetCode(status) != TF_OK) {
		std::cout << "Error delete session";
		TF_DeleteStatus(status);
		return 7;
	}

	const auto data = static_cast<float*>(TF_TensorData(output_tensor));

	std::cout << "Output vals: " << data[0] << ", " << data[1] << ", " << data[2] << ", " << data[3] << std::endl;

	
}


int realmodel()
{
	TF_Graph* graph = tf_utils::LoadGraphDef("mymodel.pb");
	if (graph == nullptr) {
		std::cout << "Can't load graph" << std::endl;
		return 1;
	}

	std::vector<TF_Output> input_op;

	input_op.push_back({ TF_GraphOperationByName(graph, "input/Placeholder"), 0 });
	if (input_op[0].oper == nullptr) {
		std::cout << "Can't init input_op" << std::endl;
		return 2;
	}	
	
	Mat img = imread("images/3DNL_record_count_1601_12_10717.jpg", IMREAD_GRAYSCALE);
	//showimage_fromMat(img);
	

	Mat test_img;

	resize(img, test_img, cv::Size(), 0.5, 0.5);
	showimage_fromMat(test_img);

	Mat base_img = test_img;

	test_img.convertTo(test_img, CV_32FC1);


	std::vector<std::int64_t> input_dims = { 1, 500, 1024, 1 };
	std::vector<float> input_vals;
	if (test_img.isContinuous()) {
		input_vals.assign((float*)test_img.datastart, (float*)test_img.dataend);
	}
	else {
		for (int i = 0; i < test_img.rows; ++i) {
			input_vals.insert(input_vals.end(), test_img.ptr<float>(i), test_img.ptr<float>(i) + test_img.cols);
		}
	}



	std::vector<TF_Tensor*> input_tensor;

	TF_Tensor * a = tf_utils::CreateTensor(TF_FLOAT,
		input_dims.data(), input_dims.size(),
		input_vals.data(), input_vals.size() * sizeof(float));
	if (a == nullptr)
	{
		cout << "error1" << endl;
		return 1;
	}
	input_tensor.push_back(a);

	
	input_op.push_back({ TF_GraphOperationByName(graph, "input/Placeholder_2"), 0 });
	if (input_op[1].oper == nullptr) {
		std::cout << "Can't init input_op" << std::endl;
		return 2;
	}
	
	std::vector<std::int64_t> input_dims2 = { 0 };
	std::vector < bool > input_vals2 = { false }; //is_training : false
	
	TF_Tensor * b = tf_utils::CreateTensor(TF_BOOL,
		input_dims2.data(), 0, //scalar value
		&input_vals2[0], input_vals2.size() * sizeof(bool));
	if (b == nullptr)
	{
		cout << "error2" << endl;
		return 1;
	}
	input_tensor.push_back(b);
	
	
	TF_Output out_op = { TF_GraphOperationByName(graph, "output/ArgMax"), 0 };
	if (out_op.oper == nullptr) {
		std::cout << "Can't init out_op" << std::endl;
		return 3;
	}
	TF_Tensor* output_tensor = nullptr;

	TF_Status* status = TF_NewStatus();
	TF_SessionOptions* options = TF_NewSessionOptions();
	TF_Session* sess = TF_NewSession(graph, options, status);
	TF_DeleteSessionOptions(options);

	if (TF_GetCode(status) != TF_OK) {
		TF_DeleteStatus(status);
		return 4;
	}


	int64 starttime = getTickCount();
	//for (int i = 0; i < 1000; i++)
	{
		TF_SessionRun(sess,
			nullptr, // Run options.
			&input_op[0], &input_tensor[0], 2, // Input tensors, input tensor values, number of inputs.
			&out_op, &output_tensor, 1, // Output tensors, output tensor values, number of outputs.
			nullptr, 0, // Target operations, number of targets.
			nullptr, // Run metadata.
			status // Output status.
		);
	}
	int64 endtime = getTickCount();
	cout << "DEEP TIME : " << ((double)endtime - starttime) / getTickFrequency() << endl;

	if (TF_GetCode(status) != TF_OK) {
		std::cout << "Error run session" << endl;
		cout << "Error code " << TF_GetCode(status) << TF_Message(status) << endl;
		TF_DeleteStatus(status);
		return 5;
	}

	TF_CloseSession(sess, status);
	if (TF_GetCode(status) != TF_OK) {
		std::cout << "Error close session";
		TF_DeleteStatus(status);
		return 6;
	}

	TF_DeleteSession(sess, status);
	if (TF_GetCode(status) != TF_OK) {
		std::cout << "Error delete session";
		TF_DeleteStatus(status);
		return 7;
	}

	cout << "End of inference" << endl;

	const auto data = static_cast<int*>(TF_TensorData(output_tensor));

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

	cout << "pred_labels : [";
	for (auto p : s)
	{
		cout << p << " ";
	}
	cout << "]" << endl;

	cv::Mat pred(500, 1024, CV_32SC1, data);
	
	pred.convertTo(pred, CV_8UC1);

	//showimage_fromMat(base_img.mul(pred, 0.5));
	showimage_fromMat(base_img.mul(pred, 0.5));

	cout << "endoffunc" << endl;

}

int main() {
	


	realmodel();
	


	
}