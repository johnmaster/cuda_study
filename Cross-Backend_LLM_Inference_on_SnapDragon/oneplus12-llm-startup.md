# OnePlus 12 本地运行 LLM 启动备忘

这份备忘记录当前已经跑通的方案：

- 手机：OnePlus 12
- 手机系统：Android 15
- 运行环境：Termux
- 推理程序：llama.cpp / llama-server
- 性能测试：llama-bench
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

### 7.1 构建 llama-bench

如果 `~/llama.cpp/build/bin/llama-bench` 已经存在，可以跳过这一步。

如果当前构建里没有 `llama-bench`，在手机 Termux 中执行：

```bash
cd ~/llama.cpp

cmake -B build \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_EXE_LINKER_FLAGS="-landroid-spawn"

cmake --build build --target llama-bench -j4
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

如果从电脑传模型到手机：

```bash
adb shell mkdir -p /sdcard/Download/models
adb push ~/models/qwen2.5-3b-instruct-q4_k_m.gguf /sdcard/Download/models/
```

## 9. 常见问题

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

cmake -B build \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_APP=OFF \
  -DLLAMA_BUILD_SERVER=ON \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_EXE_LINKER_FLAGS="-landroid-spawn"

cmake --build build --target llama-server -j4
```

### llama-server 启动后手机发热

这是正常现象。长时间运行建议：

- 插电
- 取下厚手机壳
- 先用 `-t 8`
- 上下文先用 `-c 2048`

## 10. 最短启动流程

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
