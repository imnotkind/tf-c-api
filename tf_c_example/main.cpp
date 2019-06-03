#include "haebin.h"
#include "opencv_helper.h"

int linear_model();
int pibex_model();
int fcn_model();
int fcn_model_POC();
int keras_model();
int icnet_model_POC();


int main(int argc, char** argv) {
	if (icnet_model_POC() == 1)
	{
		cout << "ERROR" << endl;
	}
}


int keras_model()
{
	cout << "keras model POC" << endl;
	const char* graph_def_filename = "models/keras/keras.pb";
	const char* checkpoint_prefix = "models/keras/keras.ckpt";
	const char* checkpoint_prefix2 = "models/keras/keras2.ckpt";
	int restore = 1;
	int frozen = 0;


	model_t model;
	cout << "Loading graph" << endl;
	if (!KERAS_ModelCreate(&model, graph_def_filename)) return 1;

	if (frozen) {
		cout << "Frozen model" << endl;
	}
	else if (restore) {
		cout << "Restoring weights from checkpoint (remove the checkpoints directory to reset)" << endl;
		if (!KERAS_ModelCheckpoint(&model, checkpoint_prefix, RESTORE)) return 1;
	}
	else {
		cout << "Initializing model weights" << endl;
		if (!KERAS_ModelInit(&model)) return 1;
	}

	vector<string> batch = { //must be same size
		"wqds_backbead_0_3.png",
		"wqds_backbead_2_8.png",
		"wqds_height_1_12.png"
	};


	cout << "Initial predictions" << endl;

	vector<Mat> img;
	for (int x = 0; x < batch.size(); x++) {
		Mat t;
		t = imread("images/" + batch[x], IMREAD_COLOR);

		cvtColor(t, t, COLOR_BGR2RGB);

		t.convertTo(t, CV_32FC3);

		if (x > 0) {
			if (img[x - 1].rows != t.rows || img[x - 1].cols != t.cols || img[x - 1].channels() != t.channels()) {
				cout << "batch img size mismatch" << endl;
				return 1;
			}
		}

		img.push_back(t);
	}

	tensor_t<float> i1; //image
	i1.dims = { (long long)img.size(), img[0].rows, img[0].cols, img[0].channels() };

	for (int x = 0; x < img.size(); x++) {
		if (img[x].isContinuous()) {
			i1.vals.insert(i1.vals.end(), (float*)img[x].datastart, (float*)img[x].dataend);
		}
		else {
			for (int i = 0; i < img[x].rows; ++i) {
				for (int j = 0; j < img[x].cols; ++j) {
					for (int k = 0; k < img[x].channels(); ++k) {
						i1.vals.push_back(img[x].at<cv::Vec3f>(i, j)[k]);
					}
				}
			}
		}
	}


	if (!KERAS_ModelPredict(&model, i1)) return 1;


	/*
	tensor_t<float> i2; //class
	i2.dims = { 1, 5 }; 
	i2.vals = { 1.0, 0.0, 0.0, 0.0, 0.0 }; //class :  0

	cout << "Training for a few steps" << endl;
	for (int i = 0; i < 50; ++i) {
		cout << "iteration " << i << endl;
		if (!KERAS_ModelRunTrainStep_POC(&model, i1, i2)) return 1;

	}

	cout << "Updated predictions" << endl;
	if (!KERAS_ModelPredict(&model, i1)) return 1;


	cout << "Saving checkpoint" << endl;
	if (!KERAS_ModelCheckpoint(&model, checkpoint_prefix2, SAVE)) return 1;
	*/


	return 0;

}

int icnet_model_POC()
{
	cout << "icnet model POC" << endl;
	const char* graph_def_filename = "models/icnet-infer.pb";
	const char* checkpoint_prefix_load = "./logs_icnet/aaa";
	const char* checkpoint_prefix_save = "./logs_icnet/bbb";
	int restore = 1;

	model_t model;
	cout << "Loading graph" << endl;
	if (!ICNET_ModelCreate(&model, graph_def_filename)) return 1;
	if (restore) {
		cout << "Restoring weights from checkpoint (remove the checkpoints directory to reset)" << endl;
		if (!ICNET_ModelCheckpoint(&model, checkpoint_prefix_load, RESTORE)) return 1;
	}
	else {
		cout << "Initializing model weights" << endl;
		if (!ICNET_ModelInit(&model)) return 1;
	}

	cout << "Initial predictions" << endl;
	Mat img = imread("images/cityscapes1.png", IMREAD_COLOR);

	showimage_fromMat(img);

	img.convertTo(img, CV_32FC3);

	tensor_t<float> i1; //image
	i1.dims = { 1024, 2048, 3 };

	
	if (img.isContinuous()) {
		i1.vals.insert(i1.vals.end(), (float*)img.datastart, (float*)img.dataend);
	}
	else {
		for (int i = 0; i < img.rows; ++i) {
			for (int j = 0; j < img.cols; ++j) {
				for (int k = 0; k < img.channels(); ++k) {
					i1.vals.push_back(img.at<cv::Vec3f>(i, j)[k]);
				}
			}
		}
	}
	
	if (!ICNET_ModelPredict(&model, i1)) return 1;
}

int fcn_model_POC()
{
	cout << "fcn model POC" << endl;
	const char* graph_def_filename = "models/fcn.pb";
	const char* checkpoint_prefix_load = "./logs_poc/model.ckpt-100000";
	const char* checkpoint_prefix_save = "./logs_poc/model.ckpt-100001";
	int restore = DirectoryExists("logs_poc");

	vector<string> batch = { //must be same size
		"Alum from Soda Cans-screenshot (5).png",
		"Alum from Soda Cans-screenshot (6).png"
	};

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


	vector<Mat> img;
	for (int x = 0; x < batch.size(); x++) {
		Mat t;
		t = imread("images/" + batch[x], IMREAD_COLOR);

		cvtColor(t, t, COLOR_BGR2RGB);

		t.convertTo(t, CV_32FC3);

		if (x > 0) {
			if (img[x - 1].rows != t.rows || img[x - 1].cols != t.cols || img[x - 1].channels() != t.channels()) {
				cout << "batch img size mismatch" << endl;
				return 1;
			}
		}

		img.push_back(t);

	}




	tensor_t<float> i1; //image
	i1.dims = { (long long)img.size(), img[0].rows, img[0].cols, img[0].channels() };

	for (int x = 0; x < img.size(); x++) {
		if (img[x].isContinuous()) {
			i1.vals.insert(i1.vals.end(), (float*)img[x].datastart, (float*)img[x].dataend);
		}
		else {
			for (int i = 0; i < img[x].rows; ++i) {
				for (int j = 0; j < img[x].cols; ++j) {
					for (int k = 0; k < img[x].channels(); ++k) {
						i1.vals.push_back(img[x].at<cv::Vec3f>(i, j)[k]);
					}
				}
			}
		}
	}


	tensor_t<float> i2; //keep_prob
	i2.dims = { 1 }; //scalar value
	i2.vals = { 1.0 }; //keep_prob : 1.0


	if (!FCN_ModelPredict(&model, i1, i2)) return 1;

	cout << "Groundtruth Label" << endl;

	vector<Mat> label;
	for (int x = 0; x < batch.size(); x++) {
		Mat l = imread("labels/" + batch[x], IMREAD_GRAYSCALE);
		show_label_image(l);
		l.convertTo(l, CV_32SC1);

		if (l.channels() != 1) {
			cout << "label channel assumed to be 1" << endl;
			return 1;
		}

		label.push_back(l);
	}


	tensor_t<int> i3; //GT_Label
	i3.dims = { (long long)label.size(), label[0].rows, label[0].cols, label[0].channels() };

	for (int x = 0; x < label.size(); x++) {
		if (label[x].isContinuous()) {
			i3.vals.insert(i3.vals.end(), (int*)label[x].datastart, (int*)label[x].dataend);
		}
		else {
			for (int i = 0; i < label[x].rows; ++i) {
				for (int j = 0; j < label[x].cols; ++j) {
					for (int k = 0; k < label[x].channels(); ++k) {
						i3.vals.push_back(label[x].at<cv::Vec3f>(i, j)[k]);
					}
				}
			}
		}
	}



	if (!FCN_ModelCalcLoss_POC(&model, i1, i2, i3)) return 1;

	cout << "Training for a few steps" << endl;
	for (int i = 0; i < 5; ++i) {
		cout << "iteration " << i << endl;
		if (!FCN_ModelRunTrainStep_POC(&model, i1, i2, i3)) return 1;
	}

	cout << "Updated predictions" << endl;
	if (!FCN_ModelPredict(&model, i1, i2)) return 1;

	if (!FCN_ModelCalcLoss_POC(&model, i1, i2, i3)) return 1;


	cout << "Saving checkpoint" << endl;
	if (!FCN_ModelCheckpoint(&model, checkpoint_prefix_save, SAVE)) return 1;


	return 0;
}

int fcn_model()
{
	cout << "fcn model" << endl;
	const char* graph_def_filename = "models/fcn.pb";
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
			for (int j = 0; j < img.cols; ++j) {
				for (int k = 0; k < img.channels(); ++k) {
					i1.vals.push_back(img.at<cv::Vec3f>(i, j)[k]);
				}
			}
		}
	}

	tensor_t<float> i2; //keep_prob
	i2.dims = {}; //scalar value
	i2.vals = { 1.0 }; //keep_prob : 1.0 when inference, 0.5 when training


	if (!FCN_ModelPredict(&model, i1, i2)) return 1;

	cout << "Training for a few steps" << endl;
	for (int i = 0; i < 50000; ++i) {
		cout << "iteration " << i << endl;
		if (!FCN_ModelRunTrainStep(&model)) return 1;

		if (i % 100 == 0)
		{
			cout << "Saving checkpoint" << endl;
			if (!FCN_ModelCheckpoint(&model, checkpoint_prefix, SAVE)) return 1;
		}
	}

	cout << "Updated predictions" << endl;
	if (!FCN_ModelPredict(&model, i1, i2)) return 1;


	cout << "Saving checkpoint" << endl;
	if (!FCN_ModelCheckpoint(&model, checkpoint_prefix, SAVE)) return 1;



	return 0;

}



int pibex_model()
{
	cout << "frozen model" << endl;
	const char* graph_def_filename = "models/mymodel.pb";

	model_t model;
	cout << "Loading graph" << endl;
	if (!PIBEX_ModelCreate(&model, graph_def_filename)) return 1;
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

	if (!PIBEX_ModelInit(&model, i1, i2)) return 1;
	
	ModelDestroy(&model);

	return 0;
}



int linear_model() {
	cout << "linear model" << endl;
	const char* graph_def_filename = "models/linear_example.pb";
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