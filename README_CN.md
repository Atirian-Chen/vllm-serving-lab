# vLLM 大模型推理服务与性能评测

这是一个刻意控制范围的单卡 vLLM 实验项目，覆盖三个适合写入 AI Infra 简历的基础机制：

- 使用 Paged KV Cache 处理混合长度请求。
- 评测 Continuous Batching 在不同并发与最大活跃序列数下的吞吐和延迟。
- 对共享长前缀流量进行 Prefix Cache 开关实验。

项目仅包含官方 vLLM OpenAI-compatible 服务、一个异步流式压测客户端和一个 PowerShell 实验编排脚本。不包含前端、数据库、Kubernetes、多卡并行、RAG 或 Agent。

## 实验矩阵

默认使用 `Qwen/Qwen2.5-1.5B-Instruct`、FP16、2,048 Token 上下文和单张 NVIDIA GPU。Chunked Prefill 保持固定镜像的默认状态，并在所有实验配置中保持一致，因此每组对照只改变目标变量。

| 配置 | 工作负载 | 客户端并发 | `max-num-seqs` | Prefix Cache |
|---|---|---:|---:|:---:|
| `cb_serial` | 128/512/1024词混合输入 | 8 | 1 | 关闭 |
| `cb_batch` | 128/512/1024词混合输入 | 8 | 8 | 关闭 |
| `cb_scale_1` | 128/512/1024词混合输入 | 1 | 8 | 关闭 |
| `cb_scale_4` | 128/512/1024词混合输入 | 4 | 8 | 关闭 |
| `pc_off` | 共享约960词前缀 | 8 | 8 | 关闭 |
| `pc_on` | 共享约960词前缀 | 8 | 8 | 开启 |

其中 `max-num-seqs=1` 是近似串行容量对照，不代表替换或删除了 vLLM 调度器。Paged KV Cache 是 vLLM 的核心机制，没有干净的关闭开关；项目通过混合长度负载验证其使用，不宣称独立的 Paged KV Cache 加速比例。

## 本机参考结果

提交结果来自 NVIDIA GeForce RTX 4070 Laptop GPU（8,188 MiB）、NVIDIA 581.57 驱动、Docker Desktop 和 `vllm/vllm-openai:v0.10.2`。18轮实验共1,080个正式请求，全部成功。下列数字均为3轮运行级指标的中位数。

| 对照 | 基线 | 优化配置 | 实测变化 |
|---|---:|---:|---:|
| 客户端并发8的输出吞吐 | `max-num-seqs=1`：9.28 tok/s | `max-num-seqs=8`：259.05 tok/s | +2690.08% |
| 客户端并发8的P95延迟 | `max-num-seqs=1`：62.09s | `max-num-seqs=8`：2.21s | -96.44% |
| 共享长前缀TTFT P50 | Prefix Cache关闭：8753ms | Prefix Cache开启：205ms | -97.66% |
| 共享长前缀有效输入吞吐 | Prefix Cache关闭：211.89 tok/s | Prefix Cache开启：4220.85 tok/s | +1891.97% |

这些数字仅代表本机固定负载。串行对照刻意将活跃序列限制为1，共享前缀实验使用约960词的重复前缀。笔记本功耗状态、WDDM和Docker Desktop造成了可见的跨轮波动，因此仓库保留全部请求记录并报告运行级中位数，不选择最快单轮。

## 指标

异步客户端通过流式 SSE 响应记录：

- TTFT P50/P95：首个非空 Token 到达时间。
- TPOT P50/P95：首 Token 后的平均单 Token 时间。
- 端到端延迟 P50/P95。
- Requests/s、有效输入 Tokens/s、输出 Tokens/s。
- 请求成功率，以及可选的 GPU 显存峰值和平均利用率。

Token 数使用服务端最终返回的 OpenAI `usage` 字段，不使用字符数或流式数据块数量代替。

正式实验默认关闭宿主机 GPU 采样，避免 Windows WDDM 驱动查询阻塞推理；如果运行环境的 `nvidia-smi` 稳定，可通过 `--gpu-sample-interval` 手动开启。

## 安装与测试

```powershell
python -m venv .venv
```

激活当前 Python 发行版创建的虚拟环境，然后运行：

```powershell
python -m pip install -e ".[dev]"
python -m pytest
```

vLLM 与 PyTorch 位于固定版本 Docker 镜像 `vllm/vllm-openai:v0.10.2` 中，不安装到 Windows 主机 Python。

## 运行实验

先启动支持 Linux 容器和 NVIDIA GPU 的 Docker Desktop：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\Run-Experiments.ps1
```

快速验证可将每组重复次数改为1：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\Run-Experiments.ps1 -Repeats 1 -RunId smoke
```

模型文件与 vLLM 编译缓存默认写入项目内的 `.cache/huggingface` 和 `.cache/vllm`，因此会保留在项目所在的 E 盘。正式实验每组包含8次预热、60次正式请求和3次独立重复，并取运行级指标中位数，生成：

```text
results/raw/<run-id>/*.json   每个请求的计时与Token记录
results/summary.csv           各配置汇总结果
results/report.md             对照结论与事实边界
```

## 简历使用边界

完成真实实验后，可以描述为“基于 vLLM 搭建单卡推理服务”“构建异步流式压测工具”“评测 Continuous Batching 与 Prefix Cache”。不能描述为“实现 PagedAttention”“实现 Continuous Batching”或“证明端到端生产性能提升”。所有数字必须附带本机硬件、模型、并发、输入输出长度和对照配置。
