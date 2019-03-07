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



# C inference

앞서 말한 lib, dll, header를 추가해서 써주면 된다.



tensorflow c api 는 사용법이 까다롭고, 할 수 있는 것도 적고, documentation이 존재하지 않아, 나 역시도 완벽하게 숙지하지는 못했다. (c_api.h를 직접 읽어야 한다.)

하지만 다행히도 인터넷에 inference의 예시로 https://github.com/Neargye/hello_tf_c_api/blob/master/src/session_run.cpp 가 있으니 참고하면서 하도록 하자.



다음은 제가 opencv를 이용해 이미지를 대상으로 inference를 돌린 코드입니다.

```c++
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


int main()
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
```





