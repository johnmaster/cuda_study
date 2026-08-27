# RTX 4090 云服务器运行和调试 vLLM Examples

本文记录一套已经验证可用的流程，用于在临时租用的 RTX 4090 云服务器上：

- 克隆 vLLM 源码，阅读 `examples/` 和 V1 实现；
- 使用与服务器驱动兼容的预编译 vLLM wheel；
- 运行 `examples/basic/offline_inference/basic.py`；
- 通过 VS Code Remote SSH 设置断点、单步执行并查看输出。

## 1. 已验证的服务器环境

```text
OS: Ubuntu 22.04
GPU: NVIDIA GeForce RTX 4090 24GB
CPU: 8 cores
RAM: 16GB
NVIDIA Driver: 550.78
nvidia-smi CUDA Version: 12.4
Python: 3.12
vLLM: 0.26.0
PyTorch/vLLM wheel variant: cu129
```

`nvidia-smi` 中的 CUDA Version 表示驱动支持能力，不表示已经安装完整
CUDA Toolkit。Driver 550 不能运行要求 `libcudart.so.13` 的 cu130 vLLM
wheel，因此必须使用 cu129 变体。

## 2. 检查基础环境

```bash
nvidia-smi
```

确认输出中包含 RTX 4090，并且驱动正常。

安装基础工具：

```bash
sudo apt update
sudo apt install -y git curl
```

安装 `uv`：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
uv --version
```

## 3. 克隆并固定 vLLM 源码版本

```bash
cd /root
git clone https://github.com/vllm-project/vllm.git
cd /root/vllm
git switch -c study-v0.26.0 v0.26.0
git describe --tags
```

预期最后一条命令显示：

```text
v0.26.0
```

源码版本必须与安装的 wheel 版本一致。学习期间不要随意在该分支执行
`git pull`，否则 examples 和运行时 API 可能不匹配。

## 4. 创建独立运行环境

不要在 `/root/vllm` 中创建或运行 Python 环境。单独建立运行目录，避免
仓库中的 `/root/vllm/vllm` 覆盖已经正确安装的包。

```bash
mkdir -p /root/vllm-run
cd /root/vllm-run

uv venv .venv-v026-cu129 \
  --python 3.12 \
  --seed \
  --managed-python
```

后续直接使用解释器绝对路径，不依赖当前激活了哪个 Conda/venv 环境：

```text
/root/vllm-run/.venv-v026-cu129/bin/python
```

## 5. 安装 cu129 vLLM wheel

先确认专用索引可以访问：

```bash
curl -fL \
  https://wheels.vllm.ai/0.26.0/cu129/vllm/metadata.json \
  | head
```

从 cu129 专用索引安装 vLLM，同时为 PyTorch 指定 cu129：

```bash
uv pip install \
  --python /root/vllm-run/.venv-v026-cu129/bin/python \
  "vllm==0.26.0" \
  --extra-index-url https://wheels.vllm.ai/0.26.0/cu129 \
  --index-strategy first-index \
  --torch-backend=cu129
```

首次安装需要下载数 GB 的 PyTorch/CUDA wheel。32Mbps 带宽下可能需要
15 到 30 分钟。`uv` 长时间停在 `Sending fresh GET request` 时通常仍在下载
大型 wheel，可以在另一个终端观察缓存：

```bash
watch -n 2 'du -sh ~/.cache/uv'
```

不要清理 `~/.cache/uv`，否则下次会重新下载。

## 6. 验证安装

必须从 `/root/vllm-run` 执行验证，不能从 `/root/vllm` 执行：

```bash
cd /root/vllm-run
```

验证 PyTorch 和 GPU：

```bash
/root/vllm-run/.venv-v026-cu129/bin/python -c \
  "import torch; print('torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

预期包含：

```text
CUDA: 12.9
available: True
GPU: NVIDIA GeForce RTX 4090
```

验证 vLLM：

```bash
/root/vllm-run/.venv-v026-cu129/bin/python -c \
  "import vllm; print('vLLM:', vllm.__version__); print('source:', vllm.__file__)"
```

源码路径应位于：

```text
/root/vllm-run/.venv-v026-cu129/lib/python3.12/site-packages/vllm/
```

如果出现 `libcudart.so.13`，说明错误安装了 cu130 vLLM wheel，需要重新按
第 5 节从 cu129 专用索引安装。仅指定 `--torch-backend=cu129` 不够，它只
控制 PyTorch，不控制 vLLM wheel。

## 7. 运行官方 example

运行时需要设置两个环境变量：

- `VLLM_ENABLE_V1_MULTIPROCESSING=0`：让 V1 EngineCore 留在当前进程，
  便于普通 Python debugger 跟踪；
- `VLLM_USE_FLASHINFER_SAMPLER=0`：避免 FlashInfer sampler 尝试调用服务器
  上不存在的 `nvcc` 进行 JIT 编译。

执行：

```bash
cd /root/vllm-run

VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
/root/vllm-run/.venv-v026-cu129/bin/python \
/root/vllm/examples/basic/offline_inference/basic.py
```

需要更多运行日志时，增加：

```bash
VLLM_LOGGING_LEVEL=DEBUG
```

例如：

```bash
VLLM_LOGGING_LEVEL=DEBUG \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
/root/vllm-run/.venv-v026-cu129/bin/python \
/root/vllm/examples/basic/offline_inference/basic.py
```

如果还需要查看 Engine 的周期性运行统计，在创建 `LLM` 时设置：

```python
llm = LLM(
    model="facebook/opt-125m",
    disable_log_stats=False,
)
```

`disable_log_stats=False` 不是环境变量，而是 `LLM(...)` 的参数。它启用的
是吞吐量、运行及等待请求数、GPU KV Cache 使用率和 prefix cache 命中率等
统计，例如：

```text
Avg prompt throughput: 123.4 tokens/s
Avg generation throughput: 45.6 tokens/s
Running: 1 reqs, Waiting: 0 reqs
GPU KV cache usage: 2.3%
Prefix cache hit rate: 75.0%
```

它与 `VLLM_LOGGING_LEVEL=DEBUG` 控制的内容不同：前者决定是否收集并输出
Engine 统计，后者决定现有的 DEBUG 级别日志是否显示。研究 Scheduler、KV
Cache 或 automatic prefix caching 时可以同时启用。短 example 可能在统计
周期触发前就执行完毕，因此即使设置为 `False`，也不保证一定出现周期统计行。

需要记录 vLLM Python 函数调用时，还可以增加：

```bash
VLLM_TRACE_FUNCTION=1
```

它会生成大量函数调用记录，并显著降低运行速度，只建议用于排查调用路径、
卡死或崩溃。日志通常保存在 `/tmp/root/vllm/vllm-instance-*/`。日常调试优先
只使用 `VLLM_LOGGING_LEVEL=DEBUG`。

同时开启 DEBUG 日志和函数调用追踪，并将终端输出保存到文件：

```bash
cd /root/vllm-run

VLLM_LOGGING_LEVEL=DEBUG \
VLLM_TRACE_FUNCTION=1 \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
/root/vllm-run/.venv-v026-cu129/bin/python \
/root/vllm/examples/basic/offline_inference/basic.py \
2>&1 | tee /root/vllm-run/basic-trace.log
```

`tee` 保存的是终端中的 vLLM 日志；逐函数 trace 仍会单独写入
`/tmp/root/vllm/vllm-instance-*/VLLM_TRACE_FUNCTION_*.log`。

第一次运行会下载 `facebook/opt-125m`。模型默认保存在：

```text
/root/.cache/huggingface
```

运行成功时会看到 FlashAttention 初始化日志和 `Generated Outputs`。
以下警告可以忽略：

```text
FutureWarning: The cuda.cudart module is deprecated
FutureWarning: The cuda.nvrtc module is deprecated
```

## 8. 调整 example 以便调试

打开：

```text
/root/vllm/examples/basic/offline_inference/basic.py
```

可以把：

```python
llm = LLM(model="facebook/opt-125m")
```

改成：

```python
llm = LLM(
    model="facebook/opt-125m",
    enforce_eager=True,
    max_model_len=1024,
    gpu_memory_utilization=0.8,
)
```

`enforce_eager=True` 会关闭 CUDA Graph，更适合观察 Python 调用路径。

## 9. 配置 VS Code Remote SSH

本机 VS Code 安装以下扩展：

- Remote - SSH；
- Python（Microsoft）；
- Python Debugger（Microsoft，扩展 ID 为 `ms-python.debugpy`）。

通过 Remote SSH 连接服务器并打开：

```text
/root/vllm
```

Python 和 Python Debugger 必须安装在远程端。扩展页面应显示类似：

```text
Installed - SSH: ubuntu22
```

如果出现 `Configured debug type 'debugpy' is not supported`，说明 Python
Debugger 只安装在本机、未安装在 SSH 远程端。安装后执行：

```text
Developer: Reload Window
```

## 10. 创建 VS Code 调试配置

先按 `Ctrl+Shift+P`，执行 `Python: Select Interpreter`。选择
`Enter interpreter path...`，输入并确认：

```text
/root/vllm-run/.venv-v026-cu129/bin/python
```

不要选择 `Create Virtual Environment`。创建：

```text
/root/vllm/.vscode/settings.json
```

内容：

```json
{
  "python.defaultInterpreterPath": "/root/vllm-run/.venv-v026-cu129/bin/python"
}
```

创建：

```text
/root/vllm/.vscode/launch.json
```

内容：

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Debug vLLM basic example",
      "type": "debugpy",
      "request": "launch",
      "python": "/root/vllm-run/.venv-v026-cu129/bin/python",
      "program": "/root/vllm/examples/basic/offline_inference/basic.py",
      "cwd": "/root/vllm-run",
      "console": "integratedTerminal",
      "justMyCode": false,
      "stopOnEntry": true,
      "subProcess": true,
      "env": {
        "CUDA_VISIBLE_DEVICES": "0",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_USE_FLASHINFER_SAMPLER": "0"
      }
    }
  ]
}
```

关键设置：

- `python` 指向已验证可用的 cu129 环境；
- `program` 使用绝对路径，不受多文件夹工作区影响；
- `cwd` 设为 `/root/vllm-run`，避免导入仓库中未安装的本地包；
- `justMyCode=false` 允许进入 site-packages 中的 vLLM Python 源码；
- `stopOnEntry=true` 便于启动后立即确认解释器和模块路径；
- `subProcess=true` 允许调试器跟踪可能产生的 Python 子进程；
- 关闭 V1 多进程后，普通断点可以进入 EngineCore 和 worker 调用链；
- 关闭 FlashInfer sampler 后，不再要求服务器安装完整 CUDA Toolkit。

保存后执行 `Developer: Reload Window`，然后按 F5。下面这种启动命令是
正确的：

```text
/usr/bin/env /root/vllm-run/.venv-v026-cu129/bin/python .../debugpy/launcher ... -- /root/vllm/examples/basic/offline_inference/basic.py
```

`/usr/bin/env` 是正常的；关键是它后面必须是虚拟环境中的 Python，而不是
`/usr/bin/python3`。

## 11. 设置断点并查看输出

先在 `examples/basic/offline_inference/basic.py` 的以下位置设置断点：

```python
llm = LLM(...)
```

```python
outputs = llm.generate(prompts, sampling_params)
```

```python
for output in outputs:
```

选择 `Debug vLLM basic example` 并按 F5。

快捷键：

- F5：继续到下一个断点；
- F10：单步执行，不进入函数；
- F11：进入函数；
- Shift+F11：退出当前函数；
- Shift+F5：停止调试。

变量在 VS Code 左侧 Variables 面板查看，模型日志和 `print()` 输出在底部
Terminal 面板查看。

### 11.1 打开 Debug Console

程序停在断点后，按 `Ctrl+Shift+Y`，或选择 `View -> Debug Console`。也可以
按 `Ctrl+Shift+P`，执行 `Debug: Focus on Debug Console View`。

确认实际解释器：

```python
import sys; sys.executable
```

预期输出：

```text
/root/vllm-run/.venv-v026-cu129/bin/python
```

确认调试环境变量：

```python
import os
os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")
os.environ.get("VLLM_USE_FLASHINFER_SAMPLER")
```

两项都应返回 `'0'`。如果 Debug Console 不能输入，通常是程序尚未暂停；先
让程序停在入口或 `basic.py` 的断点上。

### 11.2 打开工作区外的 site-packages 源码

当前真正运行的 vLLM Python 源码位于：

```text
/root/vllm-run/.venv-v026-cu129/lib/python3.12/site-packages/vllm
```

要给 vLLM 内部设置断点，可以在 VS Code 中按 `Ctrl+P` 打开以下文件：

```text
/root/vllm-run/.venv-v026-cu129/lib/python3.12/site-packages/vllm/entrypoints/llm.py
/root/vllm-run/.venv-v026-cu129/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py
/root/vllm-run/.venv-v026-cu129/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py
/root/vllm-run/.venv-v026-cu129/lib/python3.12/site-packages/vllm/v1/executor/uniproc_executor.py
/root/vllm-run/.venv-v026-cu129/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py
/root/vllm-run/.venv-v026-cu129/lib/python3.12/site-packages/vllm/v1/worker/gpu/model_runner.py
```

如果 `Ctrl+P` 不能打开绝对路径，选择 `File -> Open File...`，输入完整路径。
VS Code 当前打开 `/root/vllm` 工作区并不妨碍打开工作区外的文件，也不需要
切换工作区。

断点应放在函数体内的第一条可执行语句上，不要放在 `def`、docstring、注释
或空行上。实心红点表示断点已经绑定；空心或灰色断点通常说明源码路径和
运行时加载路径不一致。

### 11.3 给 `_run_engine` 设置断点

先在服务器终端查找它的位置：

```bash
rg -n "def _run_engine" \
  /root/vllm/vllm \
  /root/vllm-run/.venv-v026-cu129/lib/python3.12/site-packages/vllm
```

调试程序暂停后，在 Debug Console 中确认实际加载文件：

```python
import inspect
from vllm.entrypoints.llm import LLM
inspect.getfile(LLM._run_engine)
```

以 `inspect.getfile()` 返回的路径为准。通常是：

```text
/root/vllm-run/.venv-v026-cu129/lib/python3.12/site-packages/vllm/entrypoints/llm.py
```

按 `Ctrl+P` 或用 `File -> Open File...` 打开该文件，搜索
`def _run_engine`，在函数体的第一条可执行语句左侧设置断点，然后重新按
F5。

在 `/root/vllm/vllm/` 下设置内部断点不会命中，因为当前执行的是
site-packages 中的 vLLM；但 `/root/vllm/examples/` 中的断点会正常命中。

如果希望长期从资源管理器浏览运行时源码，可以选择
`File -> Add Folder to Workspace...`，添加：

```text
/root/vllm-run/.venv-v026-cu129/lib/python3.12/site-packages/vllm
```

由于 `launch.json` 使用绝对 `program` 路径，添加第二个文件夹不会改变
example 的启动位置。

## 12. 常见错误

### `libcudart.so.13` 不存在

原因：安装了 cu130 vLLM wheel，而 Driver 550 只能使用 CUDA 12 系列。

处理：重新使用第 5 节的 cu129 专用索引安装。

### `Could not find nvcc` 或 `/usr/local/cuda` 不存在

原因：FlashInfer sampler 尝试 JIT 编译，但服务器只有 CUDA runtime，没有
完整 CUDA Toolkit。

处理：运行和调试时设置：

```bash
VLLM_USE_FLASHINFER_SAMPLER=0
```

无需为了基础 example 安装完整 CUDA Toolkit。

### Hugging Face 模型下载停滞

测试：

```bash
curl -I \
  --connect-timeout 10 \
  --max-time 20 \
  https://huggingface.co/facebook/opt-125m/resolve/main/config.json
```

模型缓存位于 `/root/.cache/huggingface`。

### VS Code 不支持 `debugpy`

将 Microsoft Python Debugger 扩展安装到 SSH 远程端，然后 Reload Window。

### vLLM 内部断点没有命中

依次检查：

1. 启动命令是否使用 `.venv-v026-cu129/bin/python`；
2. `launch.json` 是否设置了 `justMyCode=false`；
3. 是否设置了 `VLLM_ENABLE_V1_MULTIPROCESSING=0`；
4. 是否在 `inspect.getfile()` 返回的 site-packages 文件中设置断点；
5. 断点是否位于函数体内的可执行语句；
6. example 是否使用了 `enforce_eager=True`，避免 CUDA Graph 绕过后续的
   Python 执行路径。

## 13. 避免下次重新下载

本次环境跑通后，在释放服务器之前执行：

```bash
du -sh /root/.cache/uv
du -sh /root/.cache/huggingface
du -sh /root/vllm-run
```

优先在云平台制作系统盘快照或自定义镜像。至少保留：

```text
/root/vllm
/root/vllm-run
/root/.cache/uv
/root/.cache/huggingface
```

推荐优先级：

```text
自定义系统镜像 > 保留系统盘 > 持久数据盘 > 每次重新下载
```

如果必须重新租用全新服务器，按照本文从第 2 节开始执行即可。
