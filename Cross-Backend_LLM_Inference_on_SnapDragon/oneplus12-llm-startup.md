# OnePlus 12 本地运行 LLM 启动备忘

这份备忘记录当前已经跑通的方案：

- 手机：OnePlus 12
- 手机系统：Android 15
- 运行环境：Termux
- 推理程序：llama.cpp / llama-server
- 性能测试：llama-bench
- 系统分析：Qualcomm Snapdragon Profiler
- 模型：Qwen2.5-3B-Instruct GGUF Q4_K_M
- 连接方式：USB-C + ADB
- 访问方式：电脑通过 `adb forward` 访问手机上的 `llama-server`

## 1. 确认 ADB 连接

电脑上执行：

```bash
adb devices
```

正常结果应类似：

```text
List of devices attached
6508962d	device
```

如果显示 `unauthorized`，手机上需要允许 USB 调试授权。

## 2. 从 Linux 连接手机 Termux

`adb shell` 进入的是 Android 系统 shell，不是 Termux 环境。要从 Linux 电脑直接操作 Termux，可以在 Termux 里启动 `sshd`，然后选择下面两种连接方式之一：

- 通过手机 IP 直接连 Termux：简单直接，但要求这个 IP 从电脑可达
- 通过 ADB 转发连 Termux：只依赖 USB 调试连接，不要求手机 IP 从电脑可达

### 2.1 手机 Termux 准备 SSH

第一次使用时，在手机 Termux 中执行：

```bash
pkg update
pkg install openssh
passwd
whoami
sshd
```

说明：

- `passwd` 用来设置 SSH 登录密码
- `whoami` 会输出 Termux 用户名，后面 SSH 登录要用
- Termux 的 `sshd` 默认监听 `8022` 端口

如果不确定 `sshd` 是否已经启动，可以在 Termux 中执行：

```bash
pgrep sshd
```

没有输出时重新启动：

```bash
sshd
```

### 2.2 方式 A：Linux 电脑直接连接手机 IP

如果手机上的网络接口 IP 从 Linux 电脑可达，可以直接连接 Termux 的 `8022` 端口。

先在手机 Termux 或 `adb shell` 里查看 IP：

```bash
ifconfig
```

例如看到：

```text
rmnet_data3: flags=65<UP,RUNNING>  mtu 1410
        inet 10.70.157.125  netmask 255.255.255.252
```

则可以在 Linux 电脑上尝试：

```bash
ssh -p 8022 <termux-username>@10.70.157.125
```

其中 `<termux-username>` 替换成手机 Termux 中 `whoami` 的输出。

注意：`rmnet_data*` 通常是手机蜂窝数据接口，`10.x.x.x` 多半是运营商内网地址，不一定能从电脑直接访问。如果电脑连不上，优先检查：

- 手机和电脑是否真的在可互通的网络里
- `sshd` 是否已在 Termux 中启动
- SSH 端口是否为 Termux 默认的 `8022`
- 手机 IP 是否变化

如果直连手机 IP 不通，使用下一种 ADB 转发方式。

### 2.3 方式 B：Linux 电脑通过 USB 转发 SSH 端口

电脑上确认 ADB 已连接后，执行：

```bash
adb forward tcp:8022 tcp:8022
```

然后从 Linux 电脑连接手机 Termux：

```bash
ssh -p 8022 <termux-username>@127.0.0.1
```

其中 `<termux-username>` 替换成手机 Termux 中 `whoami` 的输出。

连接成功后，当前终端就已经在手机 Termux 环境里了，可以直接执行 `cd ~/llama.cpp`、启动 `llama-server` 等命令。

如果需要清掉 SSH 端口转发：

```bash
adb forward --remove tcp:8022
```

这里的 `adb forward --remove tcp:8022` 只是删除电脑到手机 `8022` 端口的转发规则，不会删除手机文件或 Termux 环境。

### 2.4 可选：使用 SSH key 免密码登录

Linux 电脑上如果已经有 SSH 公钥，可以传到手机 Termux：

```bash
adb push ~/.ssh/id_rsa.pub /sdcard/Download/linux_id_rsa.pub
```

手机 Termux 中执行：

```bash
mkdir -p ~/.ssh
cat /sdcard/Download/linux_id_rsa.pub >> ~/.ssh/authorized_keys
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```

之后 Linux 电脑再连接：

```bash
ssh -p 8022 <termux-username>@127.0.0.1
```

## 3. 手机端启动 llama-server

在手机 Termux 中执行，或者按上一节从 Linux SSH 进入 Termux 后执行：

```bash
cd ~/llama.cpp

./build/bin/llama-server \
  -m /sdcard/Download/models/qwen2.5-3b-instruct-q4_k_m.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  -t 8 \
  -c 2048
```

看到类似输出即表示模型已经加载并开始监听：

```text
model loaded
server is listening on http://127.0.0.1:8080
```

注意：这里的 `127.0.0.1:8080` 是手机自己的本地地址，电脑不能直接访问，需要下一步端口转发。

## 4. 电脑端端口转发

电脑另开一个终端，执行：

```bash
adb forward tcp:18080 tcp:8080
```

含义：

```text
电脑 127.0.0.1:18080 -> 手机 127.0.0.1:8080
```

之后电脑访问：

```text
http://127.0.0.1:18080
```

如果需要清掉端口转发：

```bash
adb forward --remove tcp:18080
```

查看当前已有转发：

```bash
adb forward --list
```

## 5. 健康检查

电脑上执行：

```bash
curl http://127.0.0.1:18080/health
```

如果返回正常，说明电脑已经能通过 USB 访问手机上的 LLM 服务。

## 6. 聊天测试

电脑上执行 OpenAI 兼容接口请求：

```bash
curl http://127.0.0.1:18080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [
      {"role": "user", "content": "你好，请用中文介绍你自己。"}
    ],
    "max_tokens": 128
  }'
```

也可以使用 llama.cpp 原生 completion 接口：

```bash
curl http://127.0.0.1:18080/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt":"你好，请用中文介绍你自己。","n_predict":128}'
```

## 7. llama-bench 性能测试

`llama-bench` 用来测模型在手机上的 prompt 处理速度和生成速度，不需要启动 `llama-server`。

### 7.1 构建 llama-bench / llama-server / llama-cli

如果 `~/llama.cpp/build/bin/llama-bench`、`llama-server`、`llama-cli` 已经存在，可以跳过这一步。

当前推荐使用 `RelWithDebInfo`，性能接近 release，同时保留调试符号，后续用 `simpleperf` 看函数名会更有用。

在手机 Termux 中执行：

```bash
cd ~/llama.cpp
rm -rf build

cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_EXE_LINKER_FLAGS="-landroid-spawn"

cmake --build build --target llama-bench llama-server llama-cli -j4
```

编译完成后确认：

```bash
ls -lh build/bin/llama-bench build/bin/llama-server build/bin/llama-cli
```

### 7.2 运行 benchmark

在手机 Termux 中执行，或者从 Linux SSH 进入 Termux 后执行：

```bash
cd ~/llama.cpp

./build/bin/llama-bench \
  -m /sdcard/Download/models/qwen2.5-3b-instruct-q4_k_m.gguf \
  -p 512 \
  -n 128 \
  -t 4
```

也可以写成一行：

```bash
./build/bin/llama-bench -m /sdcard/Download/models/qwen2.5-3b-instruct-q4_k_m.gguf -p 512 -n 128 -t 4
```

参数含义：

- `-m`：模型文件路径
- `-p 512`：测试 512 token 的 prompt 处理速度
- `-n 128`：测试生成 128 token 的解码速度
- `-t 4`：使用 4 个 CPU 线程

输出里重点看两类指标：

- `pp512`：prompt processing，处理输入 prompt 的速度
- `tg128`：text generation，生成输出 token 的速度

一般更关心 `tg128` 的 `tok/s`，它更接近日常聊天时的生成速度。

### 7.3 不同线程数对比

可以分别测试 `-t 4`、`-t 6`、`-t 8`：

```bash
./build/bin/llama-bench -m /sdcard/Download/models/qwen2.5-3b-instruct-q4_k_m.gguf -p 512 -n 128 -t 4
./build/bin/llama-bench -m /sdcard/Download/models/qwen2.5-3b-instruct-q4_k_m.gguf -p 512 -n 128 -t 6
./build/bin/llama-bench -m /sdcard/Download/models/qwen2.5-3b-instruct-q4_k_m.gguf -p 512 -n 128 -t 8
```

手机上线程数不是越高越好。线程数太高可能触发降频，短时间看起来快，连续跑反而变慢。

建议测试时：

- 先关闭正在运行的 `llama-server`
- 手机保持插电
- 取下厚手机壳
- 每组测试之间稍微等手机降温

## 8. 模型文件位置

当前模型放在手机：

```text
/sdcard/Download/models/qwen2.5-3b-instruct-q4_k_m.gguf
```

### 8.1 在手机 Termux 中直接下载更多量化版本

先创建模型目录：

```bash
mkdir -p /sdcard/Download/models
cd /sdcard/Download/models
```

如果没有 `wget`：

```bash
pkg install wget
```

下载 `Q4_0`：

```bash
wget -O qwen2.5-3b-instruct-q4_0.gguf \
  https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_0.gguf
```

下载 `Q5_K_M`：

```bash
wget -O qwen2.5-3b-instruct-q5_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q5_k_m.gguf
```

下载 `Q8_0`：

```bash
wget -O qwen2.5-3b-instruct-q8_0.gguf \
  https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q8_0.gguf
```

也可以一次性下载：

```bash
cd /sdcard/Download/models

wget -O qwen2.5-3b-instruct-q4_0.gguf \
  https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_0.gguf

wget -O qwen2.5-3b-instruct-q5_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q5_k_m.gguf

wget -O qwen2.5-3b-instruct-q8_0.gguf \
  https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q8_0.gguf
```

下载完成后确认文件：

```bash
ls -lh /sdcard/Download/models/qwen2.5-3b-instruct-*.gguf
```

### 8.2 从电脑下载后传到手机

如果从电脑传模型到手机：

```bash
adb shell mkdir -p /sdcard/Download/models
adb push ~/models/qwen2.5-3b-instruct-q4_0.gguf /sdcard/Download/models/
adb push ~/models/qwen2.5-3b-instruct-q5_k_m.gguf /sdcard/Download/models/
adb push ~/models/qwen2.5-3b-instruct-q8_0.gguf /sdcard/Download/models/
```

### 8.3 启动时切换不同量化版本

启动 `llama-server` 或 `llama-bench` 时，只需要替换 `-m` 后面的模型文件路径。

例如测试 `Q8_0`：

```bash
./build/bin/llama-bench \
  -m /sdcard/Download/models/qwen2.5-3b-instruct-q8_0.gguf \
  -p 512 \
  -n 128 \
  -t 4
```

一般来说：

- `Q4_0`：文件更小，速度通常较快，质量低于 `Q4_K_M`
- `Q5_K_M`：质量比 `Q4_K_M` 更好一些，速度和内存占用也更高
- `Q8_0`：更接近原模型精度，但文件最大、内存占用最高、速度通常更慢

## 9. Qualcomm Snapdragon Profiler 辅助评测

Snapdragon Profiler 用来观察手机运行模型时的 CPU、GPU、DSP、系统 trace、线程状态等信息。它不是替代 `llama-bench` 的工具，而是配合 benchmark 使用：

- `llama-bench`：给出 `tok/s` 等推理性能结果
- Snapdragon Profiler：观察跑 benchmark 时手机系统内部发生了什么

### 9.1 下载文件

当前下载的包在 Linux 电脑：

```text
/home/lingbok/Downloads/Snapdragon_Profiler.Core.2026.4.Linux-AnyCPU.gz
```

这个文件不是 `.deb` 安装包，而是一个 gzip 压缩的 tar 包。解压后里面有 `SnapdragonProfiler/` 目录、`run_sdp.sh`、`SnapdragonProfiler.exe`、`README-Linux.txt` 等文件。

### 9.2 解压安装

可以解压到 `~/Downloads`：

```bash
cd ~/Downloads

gzip -dc Snapdragon_Profiler.Core.2026.4.Linux-AnyCPU.gz \
  > Snapdragon_Profiler.Core.2026.4.Linux-AnyCPU.tar

tar -xvf Snapdragon_Profiler.Core.2026.4.Linux-AnyCPU.tar
```

解压后进入目录：

```bash
cd ~/Downloads/SnapdragonProfiler
```

### 9.3 Linux 依赖

Profiler 的 Linux 客户端是 .NET/Mono GUI 程序，所以启动界面需要 `mono`。Mono 只是运行 Profiler 桌面客户端的运行时，不参与手机上的模型推理。

基础依赖：

```bash
sudo apt update
sudo apt install default-jre android-tools-adb libgtk-3-0 mono-complete
```

检查：

```bash
mono --version
java -version
adb version
```

Profiler 的 README 要求 Mono 6.12 或更高。如果 `run_sdp.sh` 提示版本过低，需要安装新版 Mono。

### 9.4 启动 Profiler

手机先通过 USB 连接电脑，并确认 ADB 可见：

```bash
adb devices
```

然后启动 Profiler：

```bash
cd ~/Downloads/SnapdragonProfiler
chmod +x run_sdp.sh
./run_sdp.sh
```

如果需要手动启动：

```bash
cd ~/Downloads/SnapdragonProfiler
LD_LIBRARY_PATH="$PWD:$LD_LIBRARY_PATH" mono SnapdragonProfiler.exe
```

### 9.5 用 Profiler 配合 llama-bench

推荐流程：

1. Linux 电脑启动 Snapdragon Profiler
2. Profiler 里连接 OnePlus 12
3. 开始采集 CPU、System、Perfetto 等 trace
4. 手机 Termux 或 SSH 进入 Termux 后运行 `llama-bench`
5. benchmark 跑完后停止采集
6. 对照 `llama-bench` 的 `tok/s` 和 Profiler 里的 CPU 频率、线程占用、调度、温度/功耗相关信息

例如在手机 Termux 中运行：

```bash
cd ~/llama.cpp

./build/bin/llama-bench \
  -m /sdcard/Download/models/qwen2.5-3b-instruct-q4_k_m.gguf \
  -p 512 \
  -n 128 \
  -t 6
```

也可以运行量化 sweep 脚本，让 Profiler 采集完整过程。

### 9.6 实际使用结论

当前 OnePlus 12 + Termux + `llama-bench` 场景下，Profiler 更适合做系统级观察：

- 推荐使用：`Trace`、`Realtime Performance Analysis`
- 可观察：CPU 核心频率、CPU 利用率、CPU load、内存、整体调度变化
- 不稳定：`CPU Sampling` 的函数调用栈 / 火焰图
- 当前现象：`CPU Sampling` 里只看到 `CPU Cycles`，没有暴露函数级 callstack metric

所以 Profiler 可以回答：

- 不同 `-t 2/4/6/8` 下哪些核心被用起来
- CPU 频率是否随时间下降
- 高线程数是否触发降频
- 不同量化模型运行时系统负载有什么区别

但当前环境下不适合回答：

- `llama.cpp` 哪个函数最耗时
- `ggml` 哪个 kernel 最耗时
- 函数级 self time / children time
- CPU 火焰图

函数级热点建议后续用 Android `simpleperf` 做。

### 9.7 Realtime Performance Analysis 怎么看

`Realtime Performance Analysis` 是实时曲线视图，但不会自动显示曲线，需要手动把 metric 加到中间画布。

步骤：

1. 首页选择 `Realtime Performance Analysis`
2. 左侧展开 `System`
3. 展开 `CPU Core Utilization`
4. 双击或拖动 `CPU 0 % Utilization`、`CPU 1 % Utilization` 等指标到中间区域
5. 展开 `CPU Core Frequency`
6. 双击或拖动 `CPU 0 Frequency`、`CPU 1 Frequency` 等指标到中间区域
7. 手机 Termux 中运行 `llama-bench`

建议优先添加：

- `System -> CPU Core Utilization`
- `System -> CPU Core Frequency`
- `System -> CPU Core Load`
- `System -> Memory`

如果中间什么都没有，通常不是 Profiler 没连接，而是还没有把 metric 添加到视图里。

### 9.8 Trace 怎么看

`Trace` 是采集一段时间后显示时间线结果。它不是一直滚动的实时监控。

使用方式：

1. 选择 `Capture -> New Trace`
2. 如果目标是 Termux 里的 `llama-bench`，不要在 `New App` 里选择普通 Android app
3. 可以在 `Running Apps` 或进程列表里选择 `llama-bench`
4. 选择系统级 CPU metrics
5. 开始采集
6. Termux 中运行较长时间的 `llama-bench`
7. 采集结束后查看时间线

Trace 中常看的曲线：

- 绿色：CPU Frequency
- 蓝色：CPU Utilization
- 粉色：CPU Load

如果右侧提示：

```text
The trace ended because the maximum duration expired.
```

说明采集因为达到最大时长自动结束。可以在：

```text
File -> Settings -> Capture -> Maximum Trace capture duration (ms)
```

把最大采集时间调大，例如 `120000` 表示 120 秒。

如果看到：

```text
No process metrics collected
```

说明当前主要采到了全局 CPU metrics，没有采到进程级 metrics。对本次模型评测来说，全局 CPU 频率和利用率仍然有参考价值。

### 9.9 CPU Sampling 和火焰图

`CPU Sampling` 理论上用于看函数调用栈，界面里会出现：

```text
Sampling CPU Callstacks
Children  Self  Symbol  Shared Object
```

但这次实测中，选中 `./llama-bench` 后只有：

```text
Process -> CPU -> CPU Cycles
```

没有看到 callstack / symbol / flame graph 相关 metric。点击 `Start Capture` 后，`CPU Callstacks` 和 `CPU Perf Capture` 也可能为空。

这通常不是模型没在跑，而是当前手机权限、Termux native 进程、Profiler 采样能力之间不匹配。非 root Android 上采 Termux 进程调用栈本来就不稳定。

可以用下面命令确认手机端模型进程是否存在：

```bash
adb shell ps -A | grep -i llama
adb shell pidof llama-bench
```

如果 `llama-bench` 跑得太快，可以加大测试规模：

```bash
cd ~/llama.cpp

./build/bin/llama-bench \
  -m /sdcard/Download/models/qwen2.5-3b-instruct-q4_k_m.gguf \
  -p 8192 \
  -n 2048 \
  -t 6
```

如果这样仍然没有 `CPU Callstacks`，就不要继续在 Snapdragon Profiler 里硬找火焰图。后续改用 `simpleperf` 更合适。

### 9.10 确认手机端 Profiler 服务

Profiler 连接手机后，会在手机侧启动 `sdpservice`。可以在 Linux 电脑上检查：

```bash
adb shell ps -A | grep -i sdp
```

正常会看到类似：

```text
shell  1612  ...  S sdpservice
```

这说明手机端 Profiler service 已经运行。底部显示 `Connected to PJD110` 也说明 GUI 已连接设备，但不代表某次 capture 已经成功开始。

如果需要重启手机端服务：

```bash
adb shell pkill sdpservice
```

然后重启 Snapdragon Profiler 或重新连接设备。

### 9.11 遇到的问题：Gtk-WARNING

启动时可能出现：

```text
Gtk-WARNING **: Cannot connect attribute 'active' for cell renderer class ...
```

这个通常只是 GTK# 的 warning，不一定会导致程序退出。真正需要优先处理的是后面的 fatal error。

### 9.12 遇到的问题：libSDPCore.so undefined symbol

曾遇到过：

```text
mono: symbol lookup error: /home/lingbok/Downloads/SnapdragonProfiler/libSDPCore.so: undefined symbol: _ZNSt3__113__hash_memoryEPKvm
```

这个问题不是模型问题，也不是 Termux 问题，而是 Linux 电脑上的 C++ 运行库版本不匹配。Profiler 包里的 `libSDPCore.so` 需要更新的 `libc++` / `libc++abi`，而 Ubuntu 22.04 默认可能只加载到较旧版本。

Profiler 的 README 中提到 libc++ / libc++abi 需要较新的 LLVM 版本。可按 LLVM apt 源安装 LLVM 21 相关库：

```bash
cd ~/Downloads
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 21

sudo apt install libc++-21-dev libc++abi-21-dev
```

然后用 LLVM 21 的库启动：

```bash
cd ~/Downloads/SnapdragonProfiler

LD_LIBRARY_PATH="/usr/lib/llvm-21/lib:$PWD:$LD_LIBRARY_PATH" \
  mono SnapdragonProfiler.exe
```

如果这样可以启动，再修改 `run_sdp.sh` 中的 `LD_LIBRARY_PATH`：

```bash
export LD_LIBRARY_PATH=/usr/lib/llvm-21/lib:$PWD:$LD_LIBRARY_PATH
```

之后用：

```bash
./run_sdp.sh
```

### 9.13 Profiler 和 Termux 的关系

Profiler 在 Linux 电脑上运行，通过 ADB 连接手机并采集系统信息。模型仍然是在手机 Termux 里由 `llama.cpp` 执行。

也就是说：

- Linux 电脑：运行 Snapdragon Profiler GUI
- 手机 Termux：运行 `llama-bench` 或 `llama-server`
- ADB：让 Profiler 发现和连接手机
- Mono：只负责启动 Linux 上的 Profiler GUI

当前推荐分工：

- Profiler：看系统级 CPU 频率、利用率、load、是否降频
- `llama-bench`：记录 `pp` / `tg` 的 `tok/s`
- `simpleperf`：后续用于函数级热点和火焰图

## 10. simpleperf 和 Magisk root 记录

目标：使用 Android `simpleperf` 对 Termux 中的 `llama-bench` 做函数级 CPU profiling。未 root 时，当前 OnePlus 12 的 `simpleperf` 无法使用常见采样事件；root 后 `task-clock` 已经可以运行。

### 10.1 未 root 时遇到的问题

手机中系统自带 `simpleperf`：

```bash
adb shell which simpleperf
```

输出：

```text
/system/bin/simpleperf
```

手机架构：

```bash
adb shell getprop ro.product.cpu.abi
adb shell uname -m
```

输出：

```text
arm64-v8a
aarch64
```

未 root 时，采样 `llama-bench` 失败：

```bash
adb shell pidof llama-bench

adb shell simpleperf record \
  -p <pid> \
  -g \
  --duration 30 \
  -o /data/local/tmp/llama-bench.perf.data
```

报错：

```text
Event type 'cpu-cycles' is not supported on the device
```

改用 software event 仍失败：

```bash
adb shell simpleperf record \
  -e cpu-clock \
  -f 1000 \
  -p <pid> \
  -g \
  --duration 30 \
  -o /data/local/tmp/llama-bench.perf.data

adb shell simpleperf record \
  -e task-clock \
  -f 1000 \
  -p <pid> \
  -g \
  --duration 30 \
  -o /data/local/tmp/llama-bench.perf.data
```

报错：

```text
Event type 'cpu-clock' is not supported on the device
Event type 'task-clock' is not supported on the device
```

即使 `simpleperf list` 能列出 software events：

```text
alignment-faults
context-switches
cpu-clock
cpu-migrations
emulation-faults
major-faults
minor-faults
page-faults
task-clock
```

最小测试也失败：

```bash
adb shell simpleperf stat -e task-clock -- sleep 1
adb shell simpleperf stat -e cpu-clock -- sleep 1
```

确认过系统状态：

```bash
adb shell cat /proc/sys/kernel/perf_event_paranoid
adb shell getenforce
adb shell getprop ro.debuggable
```

输出：

```text
-1
Enforcing
0
```

`perf_event_paranoid=-1` 表示 kernel 这个开关没有限制 perf，但量产系统 `ro.debuggable=0` 且 SELinux enforcing。当前现象说明非 root 环境下 simpleperf 事件采样不可用。

### 10.2 NDK 版 simpleperf 也失败

下载 Android NDK 后，推送 NDK 自带 simpleperf：

```bash
adb push ~/android/android-ndk-r28/simpleperf/bin/android/arm64/simpleperf /data/local/tmp/simpleperf
adb shell chmod +x /data/local/tmp/simpleperf
```

测试：

```bash
adb shell /data/local/tmp/simpleperf stat -e task-clock -- sleep 1
adb shell /data/local/tmp/simpleperf stat -e cpu-clock -- sleep 1
```

仍然报错：

```text
Event type 'task-clock' is not supported on the device
Event type 'cpu-clock' is not supported on the device
```

结论：不是系统自带 simpleperf 版本太旧，而是当前非 root 量产系统环境限制了 simpleperf 事件采样。

### 10.3 当前手机版本信息

root 前记录手机信息：

```bash
adb devices
adb shell getprop ro.product.model
adb shell getprop ro.product.device
adb shell getprop ro.build.version.release
adb shell getprop ro.build.fingerprint
adb shell getprop ro.build.version.incremental
adb shell getprop ro.boot.slot_suffix
adb shell ls -l /dev/block/by-name | grep -E 'init_boot|boot'
```

当时输出关键信息：

```text
model: PJD110
device: OP5929L1
Android: 16
fingerprint: OnePlus/PJD110/OP5929L1:16/BP2A.250605.015/U.21e2c9f_2d473_2cdc5:user/release-keys
incremental: U.21e2c9f_2d473_2cdc5
slot: _b
```

分区中存在：

```text
boot_a
boot_b
init_boot_a
init_boot_b
vendor_boot_a
vendor_boot_b
```

因此 Magisk root 时优先 patch `init_boot.img`，并刷当前 slot 对应的 `init_boot_b`。

### 10.4 解锁 bootloader

电脑缺少 `fastboot` 时：

```bash
sudo apt update
sudo apt install fastboot
```

手机开启：

```text
开发者选项 -> OEM unlocking
开发者选项 -> USB debugging
```

如果找不到开发者选项：

```text
设置 -> 关于本机 -> 版本信息 -> 连点版本号 / Build number
```

进入 fastboot：

```bash
adb reboot bootloader
fastboot devices
```

正常看到：

```text
6508962d fastboot
```

解锁：

```bash
fastboot flashing unlock
```

然后在手机屏幕上用音量键和电源键确认解锁。解锁会清空手机数据。

解锁后重新打开 USB debugging，并确认：

```bash
adb shell getprop ro.boot.verifiedbootstate
adb shell getprop ro.boot.flash.locked
```

输出：

```text
orange
0
```

说明 bootloader 已解锁。

### 10.5 下载匹配镜像

当前机型是 OnePlus 12 / `PJD110` / CN 版。下载页面中应选择：

```text
OnePlus 12 -- waffle
Model ID: PJD110
Release: CN
```

下载版本必须匹配当前 fingerprint：

```text
PJD110_16.0.3.500(CN01)_CN
OnePlus/PJD110/OP5929L1:16/BP2A.250605.015/U.21e2c9f_2d473_2cdc5:user/release-keys
```

在 assets 中只需要下载 root 用的 boot 镜像包：

```text
PJD110_16.0.3.500-CN01_CN-image-boot.7z
```

不需要下载很大的：

```text
image-logical.7z.001
image-logical.7z.002
image-logical.7z.003
image-logical.7z.004
```

已下载路径：

```text
/home/lingbok/Downloads/PJD110_16.0.3.500-CN01_CN-image-boot.7z
```

解压：

```bash
cd /home/lingbok/Downloads

7z x PJD110_16.0.3.500-CN01_CN-image-boot.7z \
  -oPJD110_16.0.3.500-CN01_CN-image-boot
```

如果没有 `7z`：

```bash
sudo apt install p7zip-full
```

确认 `init_boot.img`：

```bash
find /home/lingbok/Downloads/PJD110_16.0.3.500-CN01_CN-image-boot -name "init_boot.img" -ls
```

实测路径：

```text
/home/lingbok/Downloads/PJD110_16.0.3.500-CN01_CN-image-boot/init_boot.img
```

### 10.6 Magisk patch init_boot

安装 Magisk APK：

```bash
adb install /path/to/Magisk-v30.7.apk
```

如果已经安装：

```bash
adb install -r /path/to/Magisk-v30.7.apk
```

将 `init_boot.img` 传到手机：

```bash
adb push /home/lingbok/Downloads/PJD110_16.0.3.500-CN01_CN-image-boot/init_boot.img /sdcard/Download/
```

手机上打开 Magisk，点击第一张 `Magisk` 卡片右侧的：

```text
安装
```

选择：

```text
选择并修补一个文件
```

选择：

```text
/sdcard/Download/init_boot.img
```

如果文件选择器显示“最近中没有任何相符项”，说明还在最近文件视图，不是真实目录。点击蓝色文件夹 `文件管理`，进入内部存储，再进入 `Download` 选择 `init_boot.img`。

Magisk patch 成功后输出类似：

```text
Output file is written to
/storage/emulated/0/Download/magisk_patched-30700_ZDse5.img
All done!
```

拉回电脑：

```bash
adb pull /sdcard/Download/magisk_patched-30700_ZDse5.img /home/lingbok/Downloads/
```

### 10.7 刷入 patched init_boot

刷之前再次确认 slot：

```bash
adb shell getprop ro.boot.slot_suffix
```

当前是：

```text
_b
```

进入 fastboot：

```bash
adb reboot bootloader
fastboot devices
```

刷当前槽的 `init_boot_b`：

```bash
fastboot flash init_boot_b /home/lingbok/Downloads/magisk_patched-30700_ZDse5.img
fastboot reboot
```

如果当前 slot 是 `_a`，则刷：

```bash
fastboot flash init_boot_a /home/lingbok/Downloads/magisk_patched-30700_ZDse5.img
```

### 10.8 Magisk 授权 ADB Shell

开机后 Magisk 首页显示：

```text
Magisk 当前 30.7
Ramdisk 是
```

如果执行：

```bash
adb shell su -c id
```

报：

```text
Permission denied
```

通常不是 root 失败，而是 ADB Shell 没有被 Magisk 授权。

处理方式：

1. 打开 Magisk
2. 底部进入 `超级用户`
3. 找到 `Shell` 或 `com.android.shell`
4. 设置为允许

也可以用交互方式触发授权：

```bash
adb shell
su
```

手机上弹出 Magisk 授权后点允许。

最终确认：

```bash
adb shell su -c id
```

输出：

```text
uid=0(root) gid=0(root) groups=0(root) context=u:r:magisk:s0
```

说明 root 成功。

### 10.9 root 后 simpleperf 可用

解锁会清空 `/data/local/tmp`，之前推送的 NDK 版 simpleperf 会消失。因此先使用系统版：

```bash
adb shell su -c '/system/bin/simpleperf stat -e task-clock -- sleep 1'
```

root 后已经成功：

```text
Performance counter statistics:

#        count  event_name   # count / runtime
  6.552396(ms)  task-clock   # 0.006501 cpus used

Total test time: 1.007950 seconds.
```

这说明 root 后 `simpleperf` 的 `task-clock` 事件已经可以使用。

采样前先在 Termux 里让 `llama-bench` 跑久一点：

```bash
cd ~/llama.cpp

./build/bin/llama-bench \
  -m /sdcard/Download/models/qwen2.5-3b-instruct-q4_k_m.gguf \
  -p 8192 \
  -n 2048 \
  -t 6
```

电脑上查 pid 并采样：

```bash
adb shell pidof llama-bench

adb shell su -c '/system/bin/simpleperf record \
  -e task-clock \
  -f 1000 \
  -g \
  -p <pid> \
  --duration 30 \
  -o /data/local/tmp/llama-bench.perf.data'

adb shell su -c '/system/bin/simpleperf report \
  -i /data/local/tmp/llama-bench.perf.data'

adb pull /data/local/tmp/llama-bench.perf.data /home/lingbok/Downloads/
```

其中：

- `-e task-clock`：使用 root 后已验证可用的 software event
- `-f 1000`：采样频率 1000 Hz
- `-g`：采调用栈
- `--duration 30`：采样 30 秒

如果 `llama-bench` 跑完太快，可以继续加大 `-p` / `-n`，或者用脚本循环跑多次。

### 10.10 生成 simpleperf HTML 报告

如果电脑上已经下载 Android NDK，可以用 NDK 自带的 `report_html.py` 生成 HTML 报告：

```bash
python3 /home/lingbok/android/android-ndk-r28/simpleperf/report_html.py \
  -i /home/lingbok/Downloads/llama-bench.perf.data \
  -o /home/lingbok/Downloads/llama-bench-report.html
```

成功输出：

```text
Report generated at '/home/lingbok/Downloads/llama-bench-report.html'.
```

如果是在普通用户 shell，可以直接打开：

```bash
xdg-open /home/lingbok/Downloads/llama-bench-report.html
```

也可以在浏览器地址栏输入：

```text
file:///home/lingbok/Downloads/llama-bench-report.html
```

报告里重点看：

- `Flamegraph`
- `Functions`
- `Shared libraries`
- `Call graph`
- `ggml_*`
- `llama_*`
- `matmul`
- `vec_dot`
- `quantize`
- `dequantize`

如果报告中大量显示 `[unknown]` 或只有地址，说明符号解析还不够。当前已使用 `RelWithDebInfo` 编译，后续可以继续补充符号目录，或重新编译时加 frame pointer。

### 10.11 simpleperf HTML 报告遇到的问题

如果在 root shell 中执行：

```bash
python3 ~/android/android-ndk-r28/simpleperf/report_html.py \
  -i /home/lingbok/Downloads/llama-bench.perf.data \
  -o /home/lingbok/Downloads/llama-bench-report.html
```

可能报：

```text
python3: can't open file '/root/android/android-ndk-r28/simpleperf/report_html.py': [Errno 2] No such file or directory
```

原因：root shell 里 `~` 会展开成 `/root`，而 NDK 实际在：

```text
/home/lingbok/android/android-ndk-r28
```

解决：使用绝对路径：

```bash
python3 /home/lingbok/android/android-ndk-r28/simpleperf/report_html.py \
  -i /home/lingbok/Downloads/llama-bench.perf.data \
  -o /home/lingbok/Downloads/llama-bench-report.html
```

如果 HTML 已生成，但在 root shell 中执行：

```bash
xdg-open /home/lingbok/Downloads/llama-bench-report.html
```

报：

```text
mkdir: cannot create directory '/run/user/0': Permission denied
Authorization required, but no authorization protocol specified
Error: cannot open display: :0
```

原因：root 用户不能直接连接当前普通用户的桌面会话。

解决方式：

```bash
exit
xdg-open /home/lingbok/Downloads/llama-bench-report.html
```

或者仍在 root shell 时使用普通用户打开：

```bash
sudo -u lingbok xdg-open /home/lingbok/Downloads/llama-bench-report.html
```

### 10.12 simpleperf 符号质量建议

当前编译配置已经使用：

```bash
-DCMAKE_BUILD_TYPE=RelWithDebInfo
```

这样比纯 release 更适合 simpleperf 分析。如果调用栈仍然不完整，可以后续尝试重新编译时保留 frame pointer：

```bash
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_FLAGS="-fno-omit-frame-pointer" \
  -DCMAKE_CXX_FLAGS="-fno-omit-frame-pointer" \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_EXE_LINKER_FLAGS="-landroid-spawn"

cmake --build build --target llama-bench llama-server llama-cli -j4
```

然后重新采样并生成 HTML。

### 10.13 解锁后需要重建 Termux 环境

解锁 bootloader 会清空手机数据，所以 Termux、`llama.cpp`、模型文件都没了，需要重新安装。

电脑上已有 Termux APK 时：

```bash
adb install /path/to/com.termux_1022.apk
```

如果 APK 在当前目录：

```bash
adb install com.termux_1022.apk
```

覆盖安装：

```bash
adb install -r com.termux_1022.apk
```

如果签名冲突：

```bash
adb uninstall com.termux
adb install /path/to/com.termux_1022.apk
```

Termux 重新打开后，需要重新：

```bash
pkg update
pkg upgrade
termux-setup-storage
pkg install git cmake ninja clang make wget openssh libandroid-spawn
```

然后重新 clone / 编译 `llama.cpp`，并重新下载模型。

## 11. 常见问题

### `adb reverse tcp:8080 tcp:8080` 报 Address already in use

这个场景下应该使用 `adb forward`，因为服务运行在手机上，电脑要访问手机端口：

```bash
adb forward tcp:18080 tcp:8080
```

### 浏览器没有漂亮网页 UI

编译时 UI assets 可能因为 Hugging Face 下载超时没有嵌入。只要 API 可用，不影响推理服务。

### llama-server 编译时缺少 `spawn.h`

Termux 里安装兼容包：

```bash
pkg install libandroid-spawn
```

重新构建时使用：

```bash
cd ~/llama.cpp
rm -rf build

cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_EXE_LINKER_FLAGS="-landroid-spawn"

cmake --build build --target llama-bench llama-server llama-cli -j4
```

### llama-server 启动后手机发热

这是正常现象。长时间运行建议：

- 插电
- 取下厚手机壳
- 先用 `-t 8`
- 上下文先用 `-c 2048`

## 12. 最短启动流程

以后只需要三步。

Linux 电脑连接手机 Termux：

```bash
adb forward tcp:8022 tcp:8022
ssh -p 8022 <termux-username>@127.0.0.1
```

手机 Termux，或者 SSH 进入后的终端：

```bash
cd ~/llama.cpp
./build/bin/llama-server \
  -m /sdcard/Download/models/qwen2.5-3b-instruct-q4_k_m.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  -t 8 \
  -c 2048
```

电脑终端：

```bash
adb forward tcp:18080 tcp:8080
curl http://127.0.0.1:18080/health
```

然后使用：

```text
http://127.0.0.1:18080
```
