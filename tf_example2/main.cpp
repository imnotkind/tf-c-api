#include "haebin.h"
#include "opencv_helper.h"

int linear_model();
int frozen_model();
int fcn_model();
int fcn_model_POC();

int main(int argc, char** argv) {
	if (fcn_model() == 1)
	{
		cout << "ERROR" << endl;
	}
}


int fcn_model()
{
	const char* graph_def_filename = "fcn.pb";
	const char* checkpoint_prefix = "./logs/model.ckpt-haebin";
	int restore = DirectoryExists("logs");
	

	model_t model;
	cout << "Loading graph" << endl;
	if (!FCN_ModelCreate(&model, graph_def_filename)) return 1;
	if (restore) {
		cout << "Restoring weights from checkpoint (remove the checkpoints directory to reset)" << endl;
		if (!FCN_ModelCheckpoint(&model, checkpoint_prefix, RESTORE)) return 1;
	}
	else {
		cout << "Initializing model weights" << endl;
		if (!FCN_ModelInit(&model)) return 1;
	}

	cout << "Initial predictions" << endl;

	Mat img = imread("images/acl2.jpg", IMREAD_COLOR); // BGR
	//showimage_fromMat(img);

	cvtColor(img, img, COLOR_BGR2RGB);
	//showimage_fromMat(img);

	img.convertTo(img, CV_32FC3);

	tensor_t<float> i1; //image
	i1.dims = { 1, img.rows, img.cols, img.channels() };
	if (img.isContinuous()) {
		i1.vals.assign((float*)img.datastart, (float*)img.dataend);
	}
	else {
		for (int i = 0; i < img.rows; ++i) {
			i1.vals.insert(i1.vals.end(), img.ptr<float>(i), img.ptr<float>(i) + img.cols);
		}
	}

	tensor_t<float> i2; //keep_prob
	i2.dims = {}; //scalar value
	i2.vals = { 1.0 }; //keep_prob : 1.0


	if (!FCN_ModelPredict(&model, i1, i2)) return 1;

	cout << "Training for a few steps" << endl;
	for (int i = 0; i < 100; ++i) {
		cout << "iteration " << i << endl;
		if (!FCN_ModelRunTrainStep(&model)) return 1;
	}

	cout << "Updated predictions" << endl;
	if (!FCN_ModelPredict(&model, i1, i2)) return 1;


	cout << "Saving checkpoint" << endl;
	if (!FCN_ModelCheckpoint(&model, checkpoint_prefix, SAVE)) return 1;



	return 0;

}

int fcn_model_POC()
{
	const char* graph_def_filename = "fcn.pb";
	const char* checkpoint_prefix_load = "./logs_poc/model.ckpt-100000";
	const char* checkpoint_prefix_save = "./logs_poc/model.ckpt-100001";
	int restore = DirectoryExists("logs_poc");
	

	model_t model;
	cout << "Loading graph" << endl;
	if (!FCN_ModelCreate(&model, graph_def_filename)) return 1;
	if (restore) {
		cout << "Restoring weights from checkpoint (remove the checkpoints directory to reset)" << endl;
		if (!FCN_ModelCheckpoint(&model, checkpoint_prefix_load, RESTORE)) return 1;
	}
	else {
		cout << "Initializing model weights" << endl;
		if (!FCN_ModelInit(&model)) return 1;
	}


	cout << "Initial predictions" << endl;

	Mat img = imread("images/acl2.jpg", IMREAD_COLOR); // BGR
	//showimage_fromMat(img);

	cvtColor(img, img, COLOR_BGR2RGB);
	//showimage_fromMat(img);

	img.convertTo(img, CV_32FC3);

	tensor_t<float> i1; //image
	i1.dims = { 1, img.rows, img.cols, img.channels() };
	if (img.isContinuous()) {
		i1.vals.assign((float*)img.datastart, (float*)img.dataend);
	}
	else {
		for (int i = 0; i < img.rows; ++i) {
			i1.vals.insert(i1.vals.end(), img.ptr<float>(i), img.ptr<float>(i) + img.cols);
		}
	}

	tensor_t<float> i2; //keep_prob
	i2.dims = {}; //scalar value
	i2.vals = { 1.0 }; //keep_prob : 1.0


	if (!FCN_ModelPredict(&model, i1, i2)) return 1;

	cout << "Groundtruth Label" << endl;

	Mat label = imread("labels/acl2.png", IMREAD_GRAYSCALE);
	show_label_image(label);

	label.convertTo(label, CV_32SC1);


	tensor_t<int> i3; //GT_Label
	i3.dims = { 1, label.rows, label.cols, label.channels() };
	if (label.isContinuous()) {
		i3.vals.assign((int*)label.datastart, (int*)label.dataend);
	}
	else {
		for (int i = 0; i < label.rows; ++i) {
			i3.vals.insert(i3.vals.end(), label.ptr<int>(i), label.ptr<int>(i) + label.cols);
		}
	}

	cout << "Training for a few steps" << endl;
	for (int i = 0; i < 10; ++i) {
		cout << "iteration " << i << endl;
		if (!FCN_ModelRunTrainStep_POC(&model, i1, i2, i3)) return 1;
	}

	cout << "Updated predictions" << endl;
	if (!FCN_ModelPredict(&model, i1, i2)) return 1;


	cout << "Saving checkpoint" << endl;
	if (!FCN_ModelCheckpoint(&model, checkpoint_prefix_save, SAVE)) return 1;


	return 0;
}

int frozen_model()
{
	const char* graph_def_filename = "mymodel.pb";

	model_t model;
	cout << "Loading graph" << endl;
	if (!F_ModelCreate(&model, graph_def_filename)) return 1;
	cout << "Predictions" << endl;

	Mat img = imread("images/3DNL_record_count_1601_12_10717.jpg", IMREAD_GRAYSCALE);
	//showimage_fromMat(img);

	resize(img, img, cv::Size(), 0.5, 0.5);
	showimage_fromMat(img);

	img.convertTo(img, CV_32FC1);

	tensor_t<float> i1; //image
	i1.dims = { 1, 500, 1024, 1 };
	if (img.isContinuous()) {
		i1.vals.assign((float*)img.datastart, (float*)img.dataend);
	}
	else {
		for (int i = 0; i < img.rows; ++i) {
			i1.vals.insert(i1.vals.end(), img.ptr<float>(i), img.ptr<float>(i) + img.cols);
		}
	}

	tensor_t<int> i2;
	i2.dims = {}; //scalar value
	i2.vals = { 0 }; //is_training : false

	if (!F_ModelPredict(&model, i1, i2)) return 1;
	
	ModelDestroy(&model);

	return 0;
}



int linear_model() {
	const char* graph_def_filename = "linear_example.pb";
	const char* checkpoint_prefix = "./checkpoints/checkpoint";
	int restore = DirectoryExists("checkpoints");

	model_t model;
	cout << "Loading graph" << endl;
	if (!ModelCreate(&model, graph_def_filename)) return 1;
	if (restore) {
		cout << "Restoring weights from checkpoint (remove the checkpoints directory to reset)" << endl;
		if (!ModelCheckpoint(&model, checkpoint_prefix, RESTORE)) return 1;
	}
	else {
		cout << "Initializing model weights" << endl;
		if (!ModelInit(&model)) return 1;
	}

	float testdata[3] = { 1.0, 2.0, 3.0 };
	cout << "Initial predictions" << endl;
	if (!ModelPredict(&model, &testdata[0], 3)) return 1;

	cout << "Training for a few steps" << endl;
	for (int i = 0; i < 200; ++i) {
		if (!ModelRunTrainStep(&model)) return 1;
	}

	cout << "Updated predictions" << endl;
	if (!ModelPredict(&model, &testdata[0], 3)) return 1;

	cout << "Saving checkpoint" << endl;
	if (!ModelCheckpoint(&model, checkpoint_prefix, SAVE)) return 1;


	ModelDestroy(&model);
	return 0;
}