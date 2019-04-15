# Python prebuilt

`pip install --upgrade --pre tensorflow-gpu`

- 각 release에 맞는 cuda와 cudnn을 설치해야한다
- 현재 latest release는 `tensorflow 1.13.1` , 이에 맞는 건 `cuda 10.0`, `cudnn 7.x`



# C prebuilt

https://www.tensorflow.org/install/lang_c 에서 `Windows GPU only` 다운로드, 압축 해제

https://storage.googleapis.com/tensorflow 여기서 신버전들의 prebuilt를 발견할 수 있다

prebuilt를 사용하려면 빌드된 대상 CUDA 버전을 알아야한다!!! 

https://www.tensorflow.org/install/source_windows 여기서 확인 (1.12.0은 CUDA 9.0을 깔아야함, 9.2는 안 되니 주의)

`C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Visual Studio 2017\Visual Studio Tools`에 있는 개발자 명령 프롬프트를 사용

`dumpbin /exports tensorflow.dll > dumpbin.txt` 

` dumpbin.txt` 에서 name부분만 따로 빼고, 맨 위에 `EXPORTS`를 넣어서 `tensorflow.def`로 저장하자 (나의 경우는 vs code의 `Shift+Alt+Click` 기능을 사용했음 )

`dumpbin.txt` 예시

```
    ordinal hint RVA      name

          1    0 02D78C80 ??0?$MaybeStackArray@D$0CI@@icu_62@@AEAA@AEBV01@@Z
          2    1 035F9D10 ??0?$MaybeStackArray@D$0CI@@icu_62@@QEAA@$$QEAV01@@Z
          3    2 035F9D70 ??0?$MaybeStackArray@D$0CI@@icu_62@@QEAA@H@Z
          4    3 035F9DF0 ??0?$MaybeStackArray@D$0CI@@icu_62@@QEAA@XZ
          5    4 03603B80 ??0Appendable@icu_62@@QEAA@AEBV01@@Z
          6    5 03603B80 ??0Appendable@icu_62@@QEAA@XZ
```

`tensorflow.def` 예시

```
EXPORTS
??0?$MaybeStackArray@D$0CI@@icu_62@@AEAA@AEBV01@@Z
??0?$MaybeStackArray@D$0CI@@icu_62@@QEAA@$$QEAV01@@Z
??0?$MaybeStackArray@D$0CI@@icu_62@@QEAA@H@Z
??0?$MaybeStackArray@D$0CI@@icu_62@@QEAA@XZ
??0Appendable@icu_62@@QEAA@AEBV01@@Z
```

`lib /def:tensorflow.def /OUT:tensorflow.lib /MACHINE:X64`  로 `tensorflow.lib` 생성



그렇다면 3 파일이 준비가 되었을 것이다

- `tensorflow.lib` - lib 
- `tensorflow.dll` - dll 
- `c_api.h` - header 



# C build from source

https://github.com/tensorflow/tensorflow/pull/24963

https://github.com/tensorflow/tensorflow/issues/24885

아직 windows지원이 많이 미흡한 상태라서 오류가 많으니 유의

---

https://www.tensorflow.org/install/source_windows에서 `Build the pip package` 전까지의 과정을 해주자

### bazel 0.21.0 installation  

bazel 0.21.0 (너무 높은 버전을 받으면 텐서플로우가 호환이 안 된다) 을 받자

- https://docs.bazel.build/versions/master/install-windows.html

- https://docs.bazel.build/versions/master/windows.html#build-c

- 여기에 나와있는 과정을 그대로 하면 된다 

bazel, msys2, Visual C++ Build Tools 2015(내 경우는 VS2017에서 추가 옵션을 체크해서 설치했음) 설치

---

`tensoflow 1.13.1`  : https://github.com/tensorflow/tensorflow/releases

`python configure.py`

- windows에선 XLA JIT support 끄기 (아직 지원이 안 되는듯) : https://github.com/tensorflow/tensorflow/issues/24218 
- ROCm은 AMD gpu용이라고 하니 끄기
- 좋은 cpu (일단은 6세대 이상?)에선 optimization flag에 `/arch:AVX2`를 써주자 (어차피 중요한건 gpu이기 때문에 그닥 차이는 없을듯)
- 컴파일 타임 줄이는 eigen strong inline은 켜도 되는데, 만약 빌드가 실패하면 꺼보자
- CUDA는 원하는 버전을 적자 (1.13.1은 10.0이 default다), 여기서 정확하게 소수 한자리까지 적어야 나중에 dll을 찾을 수 있다
- cudnn은 7.4.2라고 해줘도 되는데 7이라고만 해도 되는듯(default)
- RTX 2080의 CUDA compute capability는 7.5이니 7.5까지 포함해주자 `3.5,7.5`

---

`bazel build --config opt //tensorflow/tools/lib_package:libtensorflow` 하면 gpu support 됨
이미 `python configure.py`에서 CUDA옵션을 줬기 때문에 `--config=cuda`를 따로 안 해도 되는듯하다. 홈페이지에 나와있는 커맨드는 예전 버전인듯.



그렇다면 3 파일이 준비가 되었을 것이다

- `liblibtensorflow.so.ifso` - lib - `bazel-bin/tensorflow/liblibtensorflow.so.ifso` 

- `libtensorflow.so ` - dll - `bazel-bin/tensorflow/libtensorflow.so`
- `c_api.h` - header - `bazel-bin/tensorflow/tools/lib_package/libtensorflow.tar.gz` 압축 해제 후 `include/tensorflow/c/c_api.h`



# Using C Api

앞서 말한 lib, dll, header를 추가해서 써주면 된다.



C Api를 이용하기 위해서는 graph definition을 protobuf(.pb) 형식으로 빼내야 한다. 또 필요한 operation이 있으면 operation의 이름, operation에 필요한 input 또는 output tensor의 shape과 type을 알고 있어야 한다.

이런 식의 예시로 말이다.

```python
import tensorflow as tf

# Batch of input and target output (1x1 matrices)
x = tf.placeholder(tf.float32, shape=[None, 1, 1], name='input')
y = tf.placeholder(tf.float32, shape=[None, 1, 1], name='target')

# Trivial linear model
y_ = tf.identity(tf.layers.dense(x, 1), name='output')

# Optimize loss
loss = tf.reduce_mean(tf.square(y_ - y), name='loss')
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss, name='train')

init = tf.global_variables_initializer()

# tf.train.Saver.__init__ adds operations to the graph to save
# and restore variables.
saver_def = tf.train.Saver().as_saver_def()

print('Run this operation to initialize variables     : ', init.name)
print('Run this operation for a train step            : ', train_op.name)
print('Feed this tensor to set the checkpoint filename: ', saver_def.filename_tensor_name)
print('Run this operation to save a checkpoint        : ', saver_def.save_tensor_name)
print('Run this operation to restore a checkpoint     : ', saver_def.restore_op_name)

# Write the graph out to a file.
with open('graph.pb', 'wb') as f:
  f.write(tf.get_default_graph().as_graph_def().SerializeToString())
```

또 checkpoint 파일로 weight를 복구해주고 싶다면, checkpoint 파일도 필요하다. (.index, .data~)

필요한 operation의 이름들은 보통 input, output, train, initializer, checkpoint save, checkpoint restore, checkpoint filename set 하는 operation이다.



그래서 결국 알아낸 operation의 이름들로 C api에서 operation들을 찾고, TF_SessionRun으로 실행하면 된다.

Tensor는 연속된 메모리와 dimension을 명시해주면 만들어낼 수 있다.



다음은 제가 구현해낸 예시입니다. 그대로 쓰시면 됩니다.

`fcn_model()`은 

- `logs`라는 디렉토리가 있으면 거기에서 restore를 한 후, prediction을 보여준 후 training을 일정량 한 후 다시 개선된 prediction을 보여줍니다.
- `logs`라는 디렉토리가 없으면 weight initialization을 한 후, prediction을 보여준 후 training을 일정량 한 후 다시 개선된 prediction을 보여줍니다.





### opencv_helper.h

```c++
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/cvdef.h>

#include <iostream>
#include <vector>
#include <set>
#include <string>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <Windows.h>

using namespace std;
using namespace cv;


class Pallete {

public:
	vector<vector<int>> color_pallete;

	Pallete() {
		//opencv : BGR!!!
		color_pallete = {
			{ 0, 0, 0 },
			{0, 0, 255},
			{255, 0, 0},
			{0, 255, 0},
		};
		
	}
};

inline string type2str(int type) {
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



inline void showimage_fromMat(Mat image)
{
	cout << type2str(image.type()) << endl;
	cout << image.size() << endl;
	if (image.empty())                      // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return;
	}
	if (image.type() != CV_8UC1 && image.type() != CV_8UC3 && image.type() != CV_8UC4)
	{
		cout << "Not an 8bit unsigned channel 1 or 3 or 4 matrix" << std::endl;
		return;
	}
	namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Display window", image);                // Show our image inside it.
	waitKey(0); // Wait for a keystroke in the window
}


//https://stackoverflow.com/questions/35993895/create-a-rgb-image-from-pixel-labels
inline void show_label_image(Mat label)
{
	if (label.empty())
	{
		cout << "pred Mat empty" << std::endl;
		return;
	}
	if (label.type() != CV_32SC1 && label.type() != CV_8UC1)
	{
		cout << "Not an 32bit signed or 8bit unsigned channel 1 matrix : " <<type2str(label.type()) << std::endl;
		return;
	}



	Pallete p;

	Mat pred2;

	if (label.type() == CV_32SC1)
		label.convertTo(pred2, CV_8UC1);
	else
		pred2 = label;


	cv::Mat draw;

	std::vector<cv::Mat> matChannels;
	cv::split(pred2, matChannels);
	matChannels.push_back(pred2);
	matChannels.push_back(pred2);
	cv::merge(matChannels, draw);


	draw.forEach<Vec3b>
	(
		[p](Vec3b &pixel, const int * position) -> void
		{
			vector<int> t;
			if (p.color_pallete.size() > pixel[0])
				t = p.color_pallete[pixel[0]];
			else
			{
				cout << "out of pallete range" << endl;
				t = p.color_pallete[0];
			}
				
			pixel[0] = t[0];
			pixel[1] = t[1];
			pixel[2] = t[2];
		}
	);


	showimage_fromMat(draw);


}
```





### haebin.h

```c++
#pragma once

#include <iostream>
#include <vector>
#include <set>
#include <string>
#include <filesystem>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <Windows.h>


#include <c_api.h>
#include "opencv_helper.h"


using namespace std;

typedef struct model_t {
	TF_Graph* graph;
	TF_Session* session;
	TF_Status* status;

	TF_Output input, target, output;
	TF_Output input2;

	TF_Operation *init_op, *train_op, *save_op, *restore_op;
	TF_Output checkpoint_file;
} model_t;

template <typename T>
struct tensor_t {
	std::vector<std::int64_t> dims;
	std::vector<T> vals;
};

enum SaveOrRestore { SAVE, RESTORE };



//FCN train + inference
int FCN_ModelCreate(model_t* model, const char* graph_def_filename);
int FCN_ModelInit(model_t* model);
int FCN_ModelCheckpoint(model_t* model, const char* checkpoint_prefix, int type);
int FCN_ModelPredict(model_t* model, tensor_t<float> i1, tensor_t<float> i2);
void FCN_ModelDestroy(model_t* model);
int FCN_ModelRunTrainStep(model_t* model);



inline int Okay(TF_Status* status) {
	if (TF_GetCode(status) != TF_OK) {
		cerr << "ERROR: " << TF_Message(status) << endl;
		return 0;
	}
	return 1;
}

inline TF_Buffer* ReadFile(const char* filename) {

	const auto f = std::fopen(filename, "rb");
	if (f == nullptr) {
		return nullptr;
	}

	std::fseek(f, 0, SEEK_END);
	const auto fsize = ftell(f);
	std::fseek(f, 0, SEEK_SET);

	if (fsize < 1) {
		std::fclose(f);
		return nullptr;
	}

	const auto data = malloc(fsize);
	std::fread(data, fsize, 1, f);
	std::fclose(f);

	TF_Buffer* ret = TF_NewBufferFromString(data, fsize);
	free(data);
	return ret;
}

inline TF_Tensor* ScalarStringTensor(const char* str, TF_Status* status) {
	size_t nbytes = 8 + TF_StringEncodedSize(strlen(str));
	TF_Tensor* t = TF_AllocateTensor(TF_STRING, NULL, 0, nbytes);
	void* data = TF_TensorData(t);
	memset(data, 0, 8);  // 8-byte offset of first string.
	TF_StringEncode(str, strlen(str), (char*)data + 8, nbytes - 8, status);
	return t;
}

inline int DirectoryExists(const char* dirname) {
	/* linux
	struct stat buf;
	return stat(dirname, &buf) == 0;
	*/
	DWORD dwAttrib = GetFileAttributesA(dirname);

	return (dwAttrib != INVALID_FILE_ATTRIBUTES &&
		(dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
}

inline vector<string> get_all_files_names_within_folder(const char* folder)
{
	string fol(folder);
	vector<string> names;
	string search_path = fol + "/*.*";
	WIN32_FIND_DATA fd;
	HANDLE hFind = ::FindFirstFile(search_path.c_str(), &fd);
	if (hFind != INVALID_HANDLE_VALUE) {
		do {
			// read all (real) files in current folder
			// , delete '!' read other 2 default folder . and ..
			if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
				names.push_back(fd.cFileName);
			}
		} while (::FindNextFile(hFind, &fd));
		::FindClose(hFind);
	}
	return names;
}
```



### fcn.cpp

```c++
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

int FCN_ModelRunTrainStep(model_t* model)
{
	string test_label_dir = "Materials_In_Vessels/LiquidSolidLabels/";
	string train_img_dir = "Materials_In_Vessels/Train_Images/";

	

	auto k = get_all_files_names_within_folder(train_img_dir.c_str());
	for (auto fname : k)
	{
		//cout << fname;

		Mat img = imread(train_img_dir + fname, IMREAD_COLOR); // BGR
		cvtColor(img, img, COLOR_BGR2RGB);
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

		Mat label = imread(test_label_dir + fname, IMREAD_GRAYSCALE);

		if (label.empty())
		{
			//cout << " : PASS" << endl;
			continue;
		}

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

		//cout << endl;

		if (!Okay(model->status))
			return 0;
	}


	return Okay(model->status);
}


void FCN_ModelDestroy(model_t* model)
{
	TF_DeleteSession(model->session, model->status);
	Okay(model->status);
	TF_DeleteGraph(model->graph);
	TF_DeleteStatus(model->status);
}
```



### main.cpp

```c++
#include "haebin.h"
#include "opencv_helper.h"


int fcn_model();

int main(int argc, char** argv) {
	if (fcn_model_POC() == 1)
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

```






# C api Tip

- Tensor의 shape과 type을 tensorboard나 python에서 `print(t.shape, t.dtype)` 등으로 확인하자. 가끔씩 `t.dtype`이 그래프에 있는 것과 다르게 출력되는것 같긴 하니(float32 -> float64같이 사소하게) 그래프를 보는게 제일 확실한 듯
- Graph op 이름도 tensorboard나 python에서 확인해서 적용
- DT_BOOL은 int 배열로 먹여줘도 잘 인식된다. 되도록 int로 주자 (vector 관련한 이슈 때문에)
- DT_FLOAT : float
- DT_INT32 : int
- DT_INT64 : int64_t





# C++ build from source

tensorflow가 windows에서 c++ api 지원을 아직 잘 안 해서 어느 정도의 hack이 필요하다.

아직 시작 단계  : <https://github.com/tensorflow/tensorflow/pull/26152>

cuda 7.0~8.0을 쓰던 과거 버전에서는 Cmake를 지원했지만, 현재 버전에서는 Cmake 지원이 끊겨 안 되고, bazel을 이용한 컴파일을 지원한다.

심지어 tensorflow가 공식으로 지원하는 C++ api는 tensorflow project를 전부 compile하면서 tensorflow 내부에 내 프로젝트를 넣어 tensorflow의 방대한 코드를 전부 컴파일해야하는 단점이 있어 배포용으로는 부적합하다.

다행히 어느 정도의 hack을 통해 shared library를 만드는 방법을 알아냈으니 그것을 쓰면 될 것 같다.

이 repo를 따른다 : https://github.com/guikarist/tensorflow-windows-build-script

이 repo에서는 tensorflow에서 지원하는 bazel build에다 추가로 윈도우에서 shared library의 형태로 쓰기 위한 패치 작업을 모아놓은 repo이다.



이 repo의 내용대로 컴파일을 했으면 해야 할 일은 bazel build의 결과에서 적절한 파일들을 include해주는 것인데, 아직까지 충분히 symbol을 다 모아놓은 static lib가 없는 상태라서 내가 필요한 symbol을 파악하고 다시 라이브러리를 빌드해야한다. (이게 무슨 소리인지는 추후에 설명)

빌드가 끝나면 c api처럼 

`bazel-bin/tensorflow/libtensorflow_cc.so -> tensorflow_cc.dll `

`bazel-bin/tensorflow/liblibtensorflow_cc.so.ifso -> tensorflow_cc.lib`

로 추가해준다.

##### include

```
source\bazel-source\external\protobuf_archive\src
source\bazel-source\external\com_google_absl
source\bazel-source\external\eigen_archive\
source\
```

```
mkdir $TensorFlowBinDir\tensorflow\lib\ -ErrorAction SilentlyContinue
Copy-Item  $TensorFlowSourceDir\bazel-bin\tensorflow\libtensorflow_cc.so $TensorFlowBinDir\tensorflow\lib\tensorflow_cc.dll -Force
Copy-Item  $TensorFlowSourceDir\bazel-bin\tensorflow\libtensorflow_cc.so.if.lib $TensorFlowBinDir\tensorflow\lib\tensorflow_cc.lib -Force

Copy-Item $TensorFlowSourceDir\tensorflow\core $TensorFlowBinDir\tensorflow\include\tensorflow\core -Recurse -Container  -Filter "*.h" -Force
Copy-Item $TensorFlowSourceDir\tensorflow\cc $TensorFlowBinDir\tensorflow\include\tensorflow\cc -Recurse -Container -Filter "*.h" -Force

Copy-Item $TensorFlowSourceDir\bazel-genfiles\tensorflow\core\ $TensorFlowBinDir\tensorflow\include_pb\tensorflow\core -Recurse -Container -Filter "*.h" -Force
Copy-Item $TensorFlowSourceDir\bazel-genfiles\tensorflow\cc $TensorFlowBinDir\tensorflow\include_pb\tensorflow\cc -Recurse -Container -Filter "*.h" -Force

# Absl includes.
Copy-Item $TensorFlowSourceDir\bazel-source\external\com_google_absl\absl $TensorFlowBinDir\absl\include\absl\ -Recurse -Container -Filter "*.h" -Force

# Eigen includes
Copy-Item $TensorFlowSourceDir\bazel-source\external\eigen_archive\ $TensorFlowBinDir\Eigen\eigen_archive -Recurse -Force
Copy-Item $TensorFlowSourceDir\third_party\eigen3 $TensorFlowBinDir\Eigen\include\third_party\eigen3\ -Recurse -Force
```

```
D:\MyUsers\Haebin\repo\tf-c-api\Lib\tensorflow-1.13.1_cc\include
D:\MyUsers\Haebin\repo\tf-c-api\Lib\tensorflow-1.13.1_cc\include_pb
D:\MyUsers\Haebin\repo\tf-c-api\Lib\tensorflow-1.13.1_cc\Eigen\include
D:\MyUsers\Haebin\repo\tf-c-api\Lib\tensorflow-1.13.1_cc\absl\include
D:\MyUsers\Haebin\repo\tf-c-api\Lib\tensorflow-1.13.1_cc\Eigen\eigen_archive
D:\MyUsers\Haebin\repo\tf-c-api\Lib\tensorflow-1.13.1_cc\proto
```

<https://github.com/node-tensorflow/node-tensorflow/blob/master/tools/install.sh>

##### preprocessor definition

```
COMPILER_MSVC
NOMINMAX
```





# TF Lite

<https://stackoverflow.com/questions/50632152/tensorflow-convert-pb-file-to-tflite-using-python>

<https://www.tensorflow.org/lite/guide/get_started>



TF_BOOL 지원 안 함 : <https://github.com/tensorflow/tensorflow/issues/20741>





# Python TF graph -> pb file

**this is not freezing**

```python
init = tf.global_variables_initializer()
saver_def = tf.train.Saver().as_saver_def()
    
print('Run this operation to initialize variables     : ', init.name)
print('Run this operation for a train step            : ', train_op.name)
print('Feed this tensor to set the checkpoint filename: ', saver_def.filename_tensor_name)
print('Run this operation to save a checkpoint        : ', saver_def.save_tensor_name)
print('Run this operation to restore a checkpoint     : ', saver_def.restore_op_name)
    
with open('fcn.pb', 'wb') as f:
    f.write(tf.get_default_graph().as_graph_def().SerializeToString())
```



# pb file + checkpoint -> frozen pb file

```bash
freeze_graph --input_graph=/tmp/mobilenet_v1_224.pb \
  --input_checkpoint=/tmp/checkpoints/mobilenet-10202.ckpt \
  --input_binary=true \
  --output_graph=/tmp/frozen_mobilenet_v1_224.pb \
  --output_node_names=MobileNetV1/Predictions/Reshape_1
```



# frozen pb file -> tflite file

```bash
tflite_convert \
  --output_file=/tmp/mobilenet_v1_1.0_224.tflite \
  --graph_def_file=/tmp/mobilenet_v1_0.50_128/frozen_graph.pb \
  --input_arrays=input \
  --output_arrays=MobilenetV1/Predictions/Reshape_1
```

```python
import tensorflow as tf

graph_def_file = "model/mymodel.pb"
input_arrays = ["input/Placeholder", "input/Placeholder_2"]
output_arrays = ["output/ArgMax"]

converter = tf.lite.TFLiteConverter.from_frozen_graph(
  graph_def_file, input_arrays, output_arrays)
tflite_model = converter.convert()

open("model/mymodel.tflite", "wb").write(tflite_model)
```



`import/`와 `:0`는 빼도 된다