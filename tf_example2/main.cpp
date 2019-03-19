#include "haebin.h"
#include "opencv_helper.h"

int linear_model();
int frozen_model();

int main(int argc, char** argv) {
	if (frozen_model() == 1)
	{
		cout << "ERROR" << endl;
	}
}


int fcn_model()
{
	const char* graph_def_filename = "fcn.pb";

	model_t model;
	printf("Loading graph\n");
	if (!FCN_ModelCreate(&model, graph_def_filename)) return 1;

}

int frozen_model()
{
	const char* graph_def_filename = "mymodel.pb";

	Mat img = imread("images/3DNL_record_count_1601_12_10717.jpg", IMREAD_GRAYSCALE);
	//showimage_fromMat(img);
	Mat test_img;

	resize(img, test_img, cv::Size(), 0.5, 0.5);
	showimage_fromMat(test_img);

	Mat base_img = test_img;

	test_img.convertTo(test_img, CV_32FC1);

	tensor_t<float> i1;
	i1.dims = { 1, 500, 1024, 1 };
	if (test_img.isContinuous()) {
		i1.vals.assign((float*)test_img.datastart, (float*)test_img.dataend);
	}
	else {
		for (int i = 0; i < test_img.rows; ++i) {
			i1.vals.insert(i1.vals.end(), test_img.ptr<float>(i), test_img.ptr<float>(i) + test_img.cols);
		}
	}
	
	tensor_t<int> i2;
	i2.dims = {}; //scalar value
	i2.vals = { 0 }; //is_training : false



	model_t model;
	printf("Loading graph\n");
	if (!F_ModelCreate(&model, graph_def_filename)) return 1;
	printf("Predictions\n");
	if (!F_ModelPredict(&model, i1, i2, base_img)) return 1;
	
	ModelDestroy(&model);

	return 0;
}



int linear_model() {
	const char* graph_def_filename = "linear_example.pb";
	const char* checkpoint_prefix = "./checkpoints/checkpoint";
	int restore = DirectoryExists("checkpoints");

	model_t model;
	printf("Loading graph\n");
	if (!ModelCreate(&model, graph_def_filename)) return 1;
	if (restore) {
		printf(
			"Restoring weights from checkpoint (remove the checkpoints directory "
			"to reset)\n");
		if (!ModelCheckpoint(&model, checkpoint_prefix, RESTORE)) return 1;
	}
	else {
		printf("Initializing model weights\n");
		if (!ModelInit(&model)) return 1;
	}

	float testdata[3] = { 1.0, 2.0, 3.0 };
	printf("Initial predictions\n");
	if (!ModelPredict(&model, &testdata[0], 3)) return 1;

	printf("Training for a few steps\n");
	for (int i = 0; i < 200; ++i) {
		if (!ModelRunTrainStep(&model)) return 1;
	}

	printf("Updated predictions\n");
	if (!ModelPredict(&model, &testdata[0], 3)) return 1;

	printf("Saving checkpoint\n");
	if (!ModelCheckpoint(&model, checkpoint_prefix, SAVE)) return 1;


	ModelDestroy(&model);
	return 0;
}