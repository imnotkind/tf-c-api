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

따로 include해야할 라이브러리들을 추출해주는 스크립트를 만들어보았다.

```powershell
Set-StrictMode -Version latest
$ErrorActionPreference = "Stop"

$tfLibDir = "$pwd\tensorflow-1.13.1_cc"
Remove-Item $tfLibDir -ErrorAction SilentlyContinue -Force -Recurse
mkdir $tfLibDir | Out-Null

$tfSourceDir = "D:\tf-win\source"

# Tensorflow lib and dll
Copy-Item  $tfSourceDir\bazel-bin\tensorflow\libtensorflow_cc.so $tfLibDir\tensorflow_cc.dll
Copy-Item  $tfSourceDir\bazel-bin\tensorflow\liblibtensorflow_cc.so.ifso $tfLibDir\tensorflow_cc.lib

# Tensorflow includes
Copy-Item $tfSourceDir\tensorflow\core $tfLibDir\include\tensorflow\core -Recurse -Filter "*.h"
Copy-Item $tfSourceDir\tensorflow\cc $tfLibDir\include\tensorflow\cc -Recurse -Filter "*.h"

Copy-Item $tfSourceDir\bazel-genfiles\tensorflow\core $tfLibDir\include_pb\tensorflow\core -Recurse -Filter "*.h"
Copy-Item $tfSourceDir\bazel-genfiles\tensorflow\cc $tfLibDir\include_pb\tensorflow\cc -Recurse -Filter "*.h"

# Protobuf includes.
Copy-Item $tfSourceDir\bazel-source\external\protobuf_archive\src\google $tfLibDir\include_proto\google -Recurse -Filter "*.h" 

# Absl includes.
Copy-Item $tfSourceDir\bazel-source\external\com_google_absl\absl $tfLibDir\include_absl\absl -Recurse -Filter "*.h" 

# Eigen includes
Copy-Item $tfSourceDir\bazel-source\external\eigen_archive\ $tfLibDir\include_eigen_archive -Recurse
Copy-Item $tfSourceDir\third_party\eigen3 $tfLibDir\include_eigen\third_party\eigen3\ -Recurse

```

##### preprocessor definition

```
COMPILER_MSVC
NOMINMAX
```



실행해보고 external symbol이 없다고 뜨면 그 심볼들을 가지고 `tf_exported_symbols_msvc.lds`에 넣고 다시 빌드하면 된다.



# TF Lite

https://www.tensorflow.org/lite/guide/get_started

TF Lite model로 바꾸기 위해서는 몇몇 operation, type등의 제약 조건이 있다.

fcn 모델의 경우에는 shape이 `scalar`인게 제약 조건에 걸려서, `(1)` shape으로 바꿔서 해결했다.

TF_BOOL 지원 안 함 : <https://github.com/tensorflow/tensorflow/issues/20741>



TF Lite는 android java api를 써서 application(apk)를 만들 수 있다.



이론 상으로는 .tflite 파일로 바꾸기만 하면 다 실행할 수 있지만, 모델 크기가 너무 크면 interpreter가 뻗어버려서 현실적으로는 불가능하다. (8 bit quantization 최적화를 적용해도 100MB 이상임)



구글에서 추천하는 모델인 모바일용으로 최적화된 deeplab으로 하면 잘 된다.

https://www.tensorflow.org/lite/models/segmentation/overview



실시간 segmentation apk 예시 : https://github.com/tantara/JejuNet



# TF Js

tensorflow js는 브라우저로 가동되므로 매우 접근성이 편리하다.

이것 역시 tfjs가 요구하는 형식으로 model을 변환해야 하며, operation이나 type등의 제약은 있지만, 이번 fcn model의 경우에는 걸리는 제약이 없어서 잘 됐다.



다만 이 경우 문제점은 역시 너무나도 큰 fcn model이다. 이론 상으로는 아무 모델이나 변환 가능하지만, 현실적으로 웹 환경에서 쓰기에 불편할 정도로 model이 커서 다운로드 받는 시간이 너무 느리다.



또 opencv같은 라이브러리의 도움이 없으므로 tensor끼리의 연산이 구현되어 있는 방식이 까다로워, image Overlay를 비교적 허접하게 할 수밖에 없었다. (현재 제가 예시로 해놓은 visualization이 부정확함) 이것은 나중에 진짜로 사용할 일이 있을때는 제대로 만들어야 할 것이다.



http://imnotkind.tk/~imnotkind/tfjs/



# Conversion

## Python TF graph -> pb file

**this is NOT freezing, so we can use this pb for training!**

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



## pb file + checkpoint -> frozen pb file

```bash
freeze_graph --input_graph=/tmp/mobilenet_v1_224.pb \
  --input_checkpoint=/tmp/checkpoints/mobilenet-10202.ckpt \
  --input_binary=true \
  --output_graph=/tmp/frozen_mobilenet_v1_224.pb \
  --output_node_names=MobileNetV1/Predictions/Reshape_1
```

```python
import sys
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib


# Freeze the graph

input_graph_path = 'model/fcn.pb'
checkpoint_path = 'model/fcn-ckpt/model.ckpt-haebin' #prefix of checkpoint, only need .index and .data-???, not .meta
input_saver_def_path = ""
input_binary = True
output_node_names = "Pred"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_frozen_graph_name = 'model/frozen_fcn.pb'
#output_optimized_graph_name = 'optimized_'+MODEL_NAME+'.pb'
clear_devices = True


freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, checkpoint_path, output_node_names,
                          restore_op_name, filename_tensor_name,
                          output_frozen_graph_name, clear_devices, "")
```



## frozen pb file -> tflite file

```bash
tflite_convert `
  --output_file=model/frozen_fcn.tflite `
  --graph_def_file=model/frozen_fcn.pb `
  --input_arrays=input_image,keep_probability `
  --output_arrays=Pred `
  --input_shapes=1,256,256,3:1 `
  --output_format=TFLITE `
  --inference_type=QUANTIZED_UINT8 `
  --std_dev_values=128,0 --mean_values=128,1 `
  --default_ranges_min=-6 --default_ranges_max=6
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

```bash
tflite_convert `
  --output_file=model/frozen_fcn.tflite `
  --graph_def_file=model/frozen_fcn.pb `
  --input_arrays=input_image `
  --output_arrays=Pred `
  --input_shapes=1,256,256,3 `
  --output_format=TFLITE `
  --inference_type=QUANTIZED_UINT8 `
  --std_dev_values=128 --mean_values=128 `
  --default_ranges_min=-6 --default_ranges_max=6
```



`import/`와 `:0`는 빼도 된다

<https://github.com/tensorflow/tensorflow/issues/23932> : scalar value는 [1]로 대체



## frozen pb -> tf js

```bash
tensorflowjs_converter `
    --input_format=tf_frozen_model `
    --output_node_names='Pred' `
    --saved_model_tags=serve `
    --output_json=true `
    frozen_fcn.pb `
    frozen_fcn.js
```

`pip install tensorflowjs==0.8.5 ` https://github.com/tensorflow/tfjs/issues/1541

pb에서 변환할 수 있는 건 구버전 뿐이다. 구버전을 받아야 한다.

오류 뜨는 경우 : `pip install numpy --upgrade` : https://stackoverflow.com/questions/54665842/when-importing-tensorflow-i-get-the-following-error-no-module-named-numpy-cor



<https://www.tensorflow.org/js/tutorials/conversion/import_saved_model>



## keras h5 -> pb

```python
from keras import backend as K
from keras.models import load_model
import tensorflow as tf

model = load_model('model/my_model.h5')

#print(model.summary())  (None,300,300,3) -> (None, 5)
print(model.input)
print(model.output)
print(model.targets)
#print(dir(model))
#print(K.learning_phase())
K.set_learning_phase(0) #0 : test, 1 : train
#print(K.learning_phase())

sess = K.get_session()

saver = tf.train.Saver()
saver.save(sess, 'keras/keras.ckpt')

sess.graph.as_default()
graph = sess.graph


saver_def = saver.as_saver_def()
print('Feed this tensor to set the checkpoint filename: ', saver_def.filename_tensor_name)
print('Run this operation to save a checkpoint        : ', saver_def.save_tensor_name)
print('Run this operation to restore a checkpoint     : ', saver_def.restore_op_name)

with open('keras/keras.pb', 'wb') as f:
    f.write(graph.as_graph_def().SerializeToString())
```

이렇게 하면 inference는 완벽한데, 문제는 training이다. 일반적으로 tensorflow model에서는 training operation을 가동하면 training이 되지만, keras의 경우 그렇지 않고 일일히 keras가 model의 곳곳을 변환하는 방식으로 set_learning_phase()함수가 이루어져 있는 것 같다. 그래서 만약 keras가 내부적으로 모델을 변환하는 방법을 안다 하더라도, 그래프 변환은 c api에서는 할 수 없는 일이라서 불가능인 것 같다. 만약 keras가 내부적으로 쓰는 train op가 있어서 그것만 실행하면 된다면, 가능할 것이다.

이슈로 올려놓은 상태 : https://github.com/tensorflow/tensorflow/issues/28681