# Python prebuilt

`pip install --upgrade --pre tensorflow-gpu`

- 각 release에 맞는 cuda와 cudnn을 설치해야한다
- 현재 latest release는 `tensorflow 1.13.1` , 이에 맞는 건 `cuda 10.0`, `cudnn 7.x`



# C prebuilt

https://www.tensorflow.org/install/lang_c 에서 `Windows GPU only` 다운로드, 압축 해제

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
- CUDA는 원하는 버전을 적자 (1.13.1은 10.0이 default다)
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





