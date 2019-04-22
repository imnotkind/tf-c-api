# This script should be executed outside repo folder of https://github.com/guikarist/tensorflow-windows-build-script.
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
Copy-Item $tfSourceDir\bazel-source\external\protobuf_archive\src $tfLibDir\include_proto -Recurse -Filter "*.h" 

# Absl includes.
Copy-Item $tfSourceDir\bazel-source\external\com_google_absl\absl $tfLibDir\include_absl\absl -Recurse -Filter "*.h" 

# Eigen includes
Copy-Item $tfSourceDir\bazel-source\external\eigen_archive\ $tfLibDir\include_eigen_archive -Recurse -Filter "*.h" 
Copy-Item $tfSourceDir\third_party\eigen3 $tfLibDir\include_eigen\third_party\eigen3\ -Recurse -Filter "*.h"

#flatbuffer include
Copy-Item $tfSourceDir\bazel-source\external\flatbuffers\include $tfLibDir\include_flat -Recurse -Filter "*.h"