# 通过源码安装 TensorFlow

本文将解释如何编译 TensorFlow 源代码为二进制文件，并通过二进制文件安装 TensorFlow，
需要注意的是，我们已经为 Linuc，Mac 和 Windows 系统提供经过测试良好，预构建好的二进制 Tensorflow 文件，
除此之外还提供 TensorFlow 的 [docker 镜像](https://hub.docker.com/r/tensorflow/tensorflow/)。
所以建议不要自己尝试构建二进制 TensorFlow 代码，除非你能熟练通过源码构建复杂程序包，并且可以解决一些在文档中没有提到的不可预测的情况。

如果上一段话没有吓退你，很高兴。这份指南将解释如何在以下操作系统上构建 TensorFlow：
*   Ubuntu
*   Mac OS X

我们官方不支持在 Windows 上构建 TensorFlow，不过，如果你不介意参考
[Bazel on Windows](https://bazel.build/versions/master/docs/windows.html)
或者
[TensorFlow CMake build](https://github.com/tensorflow/tensorflow/tree/r0.12/tensorflow/contrib/cmake).
的经验，你可以尝试在 Windows 上搭建 TensorFlow 

## 决定安装哪种类型的 TensorFlow 

你需要从以下多种类型的 TensorFlow 中选择一个安装并构建：
* **TensorFlow 仅支持 CPU**. 如果你的系统不支持 NVIDIVA 的 GPU，需要安装这个版本. 
  值得注意的是，这个版本的 TensorFlow 通常容易安装构建，所以即使你有 NVIDIA 的 GPU，我们仍然推荐你先安装这个版本。
  
* **TensorFlow 支持 GPU**. TensorFlow 程序在 GPU 上运行会明显比在 CPU 上快。
  因此，如果你的系统有 NVIDIA 的 GPU，同时你需要运行对性能要求苛刻的程序时，你就需要安装这个版本的 TensorFlow，
  不仅需要 NVIDIA 的 GPU，你的系统还需要满足 NVIDIA 软件的要求，具体描述参考以下文档：

  * @{$install_linux#NVIDIARequirements$Installing TensorFlow on Ubuntu}
  * @{$install_mac#NVIDIARequirements$Installing TensorFlow on Mac OS}


## 克隆 TensorFlow 仓库

首先从克隆 TensorFlow 仓库开始

克隆 **最新** TensorFlow 仓库, 执行以下命令:

<pre>$ <b>git clone https://github.com/tensorflow/tensorflow</b> </pre>

<code>git clone</code> 命令创建一个命名为 “tensorflow”
 的子目录。克隆完成后，你可以执行下面的命令，创建一个特定的分支（例如一个发布分支）
 After cloning, you may optionally build a
**specific branch** (such as a release branch) by invoking the
following commands:

<pre>
$ <b>cd tensorflow</b>
$ <b>git checkout</b> <i>Branch</i> # where <i>Branch</i> is the desired branch
</pre>

例如, 执行以下命令新建“r1.0”分支，替代 master 分支：

<pre>$ <b>git checkout r1.0</b></pre>

接下来你需要准备为
[Linux](#PrepareLinux)
或者
[Mac OS](#PrepareMac)准备环境。

<a name="#PrepareLinux"></a>
## 为 Linux 准备环境

在 Linux 上构建 TensorFlow 之前， 你需要在你的系统上安装以下构建工具：

  * bazel
  * TensorFlow Python 依赖
  * 可选, 为了支持 GPU 而安装的 NVIDIA 软件包 


### 安装 Bazel

如果系统之前未安装 Bazel，需要按照以下说明安装
[安装 Bazel](https://bazel.build/versions/master/docs/install.html).


### 安装 TensorFlow Python 依赖

安装 TensorFlow之前, 你必须安装以下安装包:

  * `numpy`, 一个 TensorFlow 需要安装的用于数值处理的包。
  * `dev`, 用于添加 Python 扩展包。
  * `pip`, 用于安装和管理 Python 包。
  * `wheel`, 能够让你管理 Python 的 wheel 格式的压缩包。

执行以下命令，安装 Python 2.7 

<pre>
$ <b>sudo apt-get install python-numpy python-dev python-pip python-wheel</b>
</pre>

执行以下命令，安装 Python 3.n

<pre>
$ <b>sudo apt-get install python3-numpy python3-dev python3-pip python3-wheel</b>
</pre>


### 可选项: 安装支持 GPU 的 TensorFlow 之前的一些准备条件：

如果你构建的 TensorFlow 不支持 GPU，跳过以下步骤。

必须在你的系统中安装以下 NVIDIA <i>硬件</i>：

  * 支持 CUDA 3.0或以上的 GPU。 具体参考
    [NVIDIA 文档](https://developer.nvidia.com/cuda-gpus)
    查看支持的 GPU 列表。


必须在你的系统中安装以下 NVIDIA <i>软件</i>：

  * NVIDIA's Cuda Toolkit (>= 7.0). 推荐 8.0版本.
    详细可参考
    [NVIDIA 文档](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/#axzz4VZnqTJ2A).
    确保按照文档要求添加 Cuda 相对路径 “LD_LIBRARY_PATH” 到环境变量。
    
  * 与 NVIDIA 的 Cuda 工具包匹配的驱动程序
  * cuDNN (>= v3). 推荐 5.1版本. 细节参考
    [NVIDIA 文档](https://developer.nvidia.com/cudnn),
    注意将路径添加到 `LD_LIBRARY_PATH` 环境变量。
    
最后, 你必须安装 与 CUDA 工具包匹配的 `libcupti`>= 8.0

<pre> $ <b>sudo apt-get install cuda-command-line-tools</b> </pre>

添加路径到 `LD_LIBRARY_PATH` 环境变量:

<pre> $ <b>export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64</b> </pre>

如果 Cuda Toolkit <= 7.5, 通过调用下面的命令安装 `libcupti-dev`:

<pre> $ <b>sudo apt-get install libcupti-dev</b> </pre>


### 接下来

环境搭建好之后，你必须参考
[安装指南](#ConfigureInstallation).


<a name="PrepareMac"></a>
## Mac OS 安装准备

 构建 TensorFlow 之前, 须在你的系统上安装以下工具:

  * bazel
  * TensorFlow Python 依赖.
  * 可选项：NVIDIA 工具包（支持 GPU 的 TensorFlow 版本）.


### 安装 bazel

如果系统没有安装 Bazel ，参考
[指导](https://bazel.build/versions/master/docs/install.html#mac-os-x)安装 Bazel.


### 安装 python 依赖

安装 TensorFlow，需要安装以下依赖饱：

  * six
  * numpy, 一个 TensorFlow 需要的用于数值处理的包.
  * wheel, 能够让你管理 Python 的 wheel 格式的压缩包。

你可以通过 pip 安装 Python 依赖，如果机器上没有 pip，我们推荐使用 homebrew 去安装 Python 以及 pip，参考

[文档](http://docs.python-guide.org/en/latest/starting/install/osx/)进行安装.
如果按照以上介绍安装，将不需要禁用 SIP。

安装完 pip,调用以下命令 :

<pre> $ <b>sudo pip install six numpy wheel</b> </pre>



### 可选项: 安装支持 GPU 的 TensorFlow 时需要的前提条件 

如果你没有安装 brew，可以参考
[指导](http://brew.sh/).

安装完 brew, 按照以下命令安装 GNU 工具:

<pre>$ <b>brew install coreutils</b></pre>

如果你想编译 TensorFlow 而且安装的是 XCode 7.3 以及 CUDA 7.5，那么请注意 XCode 7.3 不能兼容 CUDA 7.5.为了弥补这个问题可以参考以下步骤

  * 更新 CUDA 到 8.0.
  * 下载 Xcode 7.2 并执行以下命令，设置其作为默认的编辑器:

    <pre> $ <b>sudo xcode-select -s /Application/Xcode-7.2/Xcode.app</b></pre>

**注意:** 你的系统需要支持 NVIDIA 软件需求，具体参考以下文档


  * @{$install_linux#NVIDIARequirements$Installing TensorFlow on Linux}
  * @{$install_mac#NVIDIARequirements$Installing TensorFlow on Mac OS}


<a name="ConfigureInstallation"></a>
## 配置安装

在文件夹根目录里有一个命名为<code>configure</code> 的 bash 脚本。
这个脚本会要求你定义与 TensorFlow 相关依赖路径以及指定其他相关的配置选项，例如编译器标记。
你必须在创建 pip 包以及安装 TensorFlow *之前*运行这个脚本。

如果你希望构建的 TensorFlow 支持 GPU，`configure`将会要求你指明安装在系统上的 Cuda 以及 cuDNN 的版本，
明确选择期望的版本取代默认选项。

`configure` 将会询问以下内容:

<pre>
Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]
</pre>

这里指的是可以在后面指定你用来[构建 pip 安装包](#build-the-pip-package) 的 Bazel 方式。
我们推荐使用默认选项(`-march=native`)，这个
会根据你本地机器的 CPU 类型优化生成的代码，需要参考
如果你正在构建的 TensorFlow 的 CPU 类型与将要运行的 CPU 类型不同，需要参考[the gcc
documentation](https://gcc.gnu.org/onlinedocs/gcc-4.5.3/gcc/i386-and-x86_002d64-Options.html)进一步的优化。

这里展示一个运行 `configure` 脚本的例子，注意你自己的输入可能不同于例子中的输入


<pre>
$ <b>cd tensorflow</b>  # cd to the top-level directory created
$ <b>./configure</b>
Please specify the location of python. [Default is /usr/bin/python]: <b>/usr/bin/python2.7</b>
Found possible Python library paths:
  /usr/local/lib/python2.7/dist-packages
  /usr/lib/python2.7/dist-packages
Please input the desired Python library path to use.  Default is [/usr/lib/python2.7/dist-packages]

Using python library path: /usr/local/lib/python2.7/dist-packages
Do you wish to build TensorFlow with MKL support? [y/N]
No MKL support will be enabled for TensorFlow
Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]:
Do you wish to use jemalloc as the malloc implementation? [Y/n]
jemalloc enabled
Do you wish to build TensorFlow with Google Cloud Platform support? [y/N]
No Google Cloud Platform support will be enabled for TensorFlow
Do you wish to build TensorFlow with Hadoop File System support? [y/N]
No Hadoop File System support will be enabled for TensorFlow
Do you wish to build TensorFlow with the XLA just-in-time compiler (experimental)? [y/N]
No XLA support will be enabled for TensorFlow
Do you wish to build TensorFlow with VERBS support? [y/N]
No VERBS support will be enabled for TensorFlow
Do you wish to build TensorFlow with OpenCL support? [y/N]
No OpenCL support will be enabled for TensorFlow
Do you wish to build TensorFlow with CUDA support? [y/N] <b>Y</b>
CUDA support will be enabled for TensorFlow
Do you want to use clang as CUDA compiler? [y/N]
nvcc will be used as CUDA compiler
Please specify the Cuda SDK version you want to use, e.g. 7.0. [Leave empty to default to CUDA 8.0]: <b>8.0</b>
Please specify the location where CUDA 8.0 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:
Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]:
Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 6.0]: <b>6</b>
Please specify the location where cuDNN 6 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:
Please specify a list of comma-separated Cuda compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
Please note that each additional compute capability significantly increases your build time and binary size.
[Default is: "3.5,5.2"]: <b>3.0</b>
Do you wish to build TensorFlow with MPI support? [y/N] 
MPI support will not be enabled for TensorFlow
Configuration finished
</pre>

如果你告知 `configure` 去支持 GPU ， `configure` 将会创建一个规范的符号链接你系统上的 Cuda 库，
因此每次你改变 Cuda 库路径时候你必须在重新调用 <code>bazel build</code> 命令前运行 `configure` 脚本。

on your system.  Therefore, every time you change the Cuda library paths,

注意:

  * Although it is possible to build both Cuda and non-Cuda configs
    under the same source tree, we recommend running `bazel clean` when
    switching between these two configurations in the same source tree.
  * If you don't run the `configure` script *before* running the
    `bazel build` command, the `bazel build` command will fail.


## Build the pip package

To build a pip package for TensorFlow with CPU-only support,
you would typically invoke the following command:

<pre>
$ <b>bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package</b>
</pre>

To build a pip package for TensorFlow with GPU support,
invoke the following command:

<pre>$ <b>bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package</b> </pre>

**NOTE on gcc 5 or later:** the binary pip packages available on the
TensorFlow website are built with gcc 4, which uses the older ABI. To
make your build compatible with the older ABI, you need to add
`--cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"` to your `bazel build` command.
ABI compatibility allows custom ops built against the TensorFlow pip package
to continue to work against your built package.

<b>Tip:</b> By default, building TensorFlow from sources consumes
a lot of RAM.  If RAM is an issue on your system, you may limit RAM usage
by specifying <code>--local_resources 2048,.5,1.0</code> while
invoking `bazel`.

The <code>bazel build</code> command builds a script named
`build_pip_package`.  Running this script as follows will build
a `.whl` file within the `/tmp/tensorflow_pkg` directory:

<pre>
$ <b>bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg</b>
</pre>


## Install the pip package

Invoke `pip install` to install that pip package.
The filename of the `.whl` file depends on your platform.
For example, the following command will install the pip package

for TensorFlow 1.4.0rc0 on Linux:

<pre>
$ <b>sudo pip install /tmp/tensorflow_pkg/tensorflow-1.4.0rc0-py2-none-any.whl</b>
</pre>

## Validate your installation

Validate your TensorFlow installation by doing the following:

Start a terminal.

Change directory (`cd`) to any directory on your system other than the
`tensorflow` subdirectory from which you invoked the `configure` command.

Invoke python:

<pre>$ <b>python</b></pre>

Enter the following short program inside the python interactive shell:

```python
# Python
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
```

If the system outputs the following, then you are ready to begin writing
TensorFlow programs:

<pre>Hello, TensorFlow!</pre>

If you are new to TensorFlow, see @{$get_started/get_started$Getting Started with
TensorFlow}.

If the system outputs an error message instead of a greeting, see [Common
installation problems](#common_installation_problems).

## Common installation problems

The installation problems you encounter typically depend on the
operating system.  See the "Common installation problems" section
of one of the following guides:

  * @{$install_linux#CommonInstallationProblems$Installing TensorFlow on Linux}
  * @{$install_mac#CommonInstallationProblems$Installing TensorFlow on Mac OS}
  * @{$install_windows#CommonInstallationProblems$Installing TensorFlow on Windows}

Beyond the errors documented in those two guides, the following table
notes additional errors specific to building TensorFlow.  Note that we
are relying on Stack Overflow as the repository for build and installation
problems.  If you encounter an error message not listed in the preceding
two guides or in the following table, search for it on Stack Overflow.  If
Stack Overflow doesn't show the error message, ask a new question on
Stack Overflow and specify the `tensorflow` tag.

<table>
<tr> <th>Stack Overflow Link</th> <th>Error Message</th> </tr>

<tr>
  <td><a
  href="https://stackoverflow.com/questions/41293077/how-to-compile-tensorflow-with-sse4-2-and-avx-instructions">41293077</a></td>
  <td><pre>W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow
  library wasn't compiled to use SSE4.1 instructions, but these are available on
  your machine and could speed up CPU computations.</pre></td>
</tr>

<tr>
  <td><a href="http://stackoverflow.com/q/42013316">42013316</a></td>
  <td><pre>ImportError: libcudart.so.8.0: cannot open shared object file:
  No such file or directory</pre></td>
</tr>

<tr>
  <td><a href="http://stackoverflow.com/q/42013316">42013316</a></td>
  <td><pre>ImportError: libcudnn.5: cannot open shared object file:
  No such file or directory</pre></td>
</tr>

<tr>
  <td><a href="http://stackoverflow.com/q/35953210">35953210</a></td>
  <td>Invoking `python` or `ipython` generates the following error:
  <pre>ImportError: cannot import name pywrap_tensorflow</pre></td>
</tr>
</table>

## Tested source configurations
**Linux**
<table>
<tr><th>Version:</th><th>CPU/GPU:</th><th>Python Version:</th><th>Compiler:</th><th>Build Tools:</th><th>cuDNN:</th><th>CUDA:</th></tr>
<tr><td>tensorflow-1.4.0rc0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.4.5</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.4.0rc0</td><td>GPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.4.5</td><td>6</td><td>8</td></tr>
<tr><td>tensorflow-1.2.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.4.5</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.2.0</td><td>GPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.4.5</td><td>5.1</td><td>8</td></tr>
<tr><td>tensorflow-1.1.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.4.2</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.1.0</td><td>GPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.4.2</td><td>5.1</td><td>8</td></tr>
<tr><td>tensorflow-1.0.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.4.2</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.0.0</td><td>GPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.4.2</td><td>5.1</td><td>8</td></tr>
</table>

**Mac**
<table>
<tr><th>Version:</th><th>CPU/GPU:</th><th>Python Version:</th><th>Compiler:</th><th>Build Tools:</th><th>cuDNN:</th><th>CUDA:</th></tr>
<tr><td>tensorflow-1.4.0rc0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>Clang from xcode</td><td>Bazel 0.4.5</td><td>N/A</td><td>N/A</td></tr>
<tr><td>ttensorflow-1.2.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>Clang from xcode</td><td>Bazel 0.4.5</td><td>N/A</td><td>N/A</td></tr>
<tr><td>ttensorflow-1.1.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>Clang from xcode</td><td>Bazel 0.4.2</td><td>N/A</td><td>N/A</td></tr>
<tr><td>ttensorflow_gpu-1.1.0</td><td>GPU</td><td>2.7, 3.3-3.6</td><td>Clang from xcode</td><td>Bazel 0.4.2</td><td>5.1</td><td>8</td></tr>
<tr><td>ttensorflow-1.0.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>Clang from xcode</td><td>Bazel 0.4.2</td><td>N/A</td><td>N/A</td></tr>
<tr><td>ttensorflow_gpu-1.0.0</td><td>GPU</td><td>2.7, 3.3-3.6</td><td>Clang from xcode</td><td>Bazel 0.4.2</td><td>5.1</td><td>8</td></tr>
</table>

**Windows**
<table>
<tr><th>Version:</th><th>CPU/GPU:</th><th>Python Version:</th><th>Compiler:</th><th>Build Tools:</th><th>cuDNN:</th><th>CUDA:</th></tr>
<tr><td>tensorflow-1.4.0rc0</td><td>CPU</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.4.0rc0</td><td>GPU</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>6</td><td>8</td></tr>
<tr><td>tensorflow-1.2.0</td><td>CPU</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.2.0</td><td>GPU</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>5.1</td><td>8</td></tr>
<tr><td>tensorflow-1.1.0</td><td>CPU</td><td>3.5</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.1.0</td><td>GPU</td><td>3.5</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>5.1</td><td>8</td></tr>
<tr><td>tensorflow-1.0.0</td><td>CPU</td><td>3.5</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.0.0</td><td>GPU</td><td>3.5</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>5.1</td><td>8</td></tr>
</table>
