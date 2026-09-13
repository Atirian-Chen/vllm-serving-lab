# vLLM 大模型推理服务与性能评测

这是一个刻意控制范围的单卡 vLLM 实验项目，覆盖四个推理服务评测方向：

- 使用 Paged KV Cache 处理混合长度请求。
- 评测 Continuous Batching 在不同并发与最大活跃序列数下的吞吐和延迟。
- 对共享长前缀流量进行 Prefix Cache 开关实验。
- 在联合 TTFT/TPOT 延迟约束下，通过固定到达率压测分析容量边界。

项目仅包含官方 vLLM OpenAI-compatible 服务、一个共享的异步流式压测客户端和 PowerShell 实验编排脚本。不包含前端、数据库、Kubernetes、多卡并行、RAG 或 Agent。

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

### 延迟约束下的容量结果（2026-09-07）

新增开环压测共 **9 轮、5,400 个正式请求**，每轮发送窗口约 120 秒，全部请求成功且发压有效。联合 SLO 要求至少 95% 的请求同时满足 TTFT ≤1000 ms、TPOT ≤50 ms/token。

| 到达率 req/s | 通过轮次 | 联合达标率范围 | TTFT P95 中位数 ms | goodput 中位数 req/s |
|---:|---:|---:|---:|---:|
| 4 | 3/3 | 100% | 123.77 | 3.969 |
| 5 | 3/3 | 100% | 335.97 | 4.958 |
| 6 | 0/3 | 3.33%–4.58% | 20,376.50 | 0.231 |

**5 req/s 是三个已测档位中、三轮全部达标的最高档；6 req/s 三轮均不达标。** 两者之间尚未细扫。6 req/s 的所有 SLO 违约均来自 TTFT，TPOT 没有超限。从 5 增到 6 req/s，输出吞吐中位数仅从 317.33 增至 323.60 tok/s，但客户端积压明显增长，排空时间中位数从 1.23 秒升至 22.57 秒。因此请求成功率或输出吞吐不能单独代表可用容量。

本次实际输入为 154/538/1050 tokens，输出为 64 tokens，`max-num-seqs=8`，Prefix Cache 关闭，显存利用率参数为 **0.50**。结论只对应本机、固定负载和有限测量窗口，不代表生产容量，也不与历史显存参数 0.85 的实验宣称前后提升。完整数据见[总报告](results/report.md#slo-constrained-capacity-experiment-2026-09-07)及[逐轮报告](results/capacity/20260907-capacity/report.md)。

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

## 有延迟约束的容量压测

```powershell
.\scripts\Run-Capacity.ps1 -Rates 4,5,6 -Offline
```

首次运行、尚未缓存模型时去掉 `-Offline`。脚本使用项目内现有 `.venv`，启动固定配置的 GPU 容器，结束后关闭该容器。`-UseExistingServer` 可以复用名为 `vllm-serving-lab-capacity` 的现有容器，并在结束后保留服务。模型、编译缓存和临时目录都位于项目所在盘。

本次扩展只增加一个容量压测模块和一个入口脚本，复用现有 SSE 客户端与混合长度负载，没有新增依赖。测量口径如下：

- **开环发压**：按固定 req/s 均匀安排请求，不等待前一请求完成；保留计划时间、实际发送时间和完成时间。
- **固定负载**：128/512/1024 词正文约各占三分之一，另有请求标识和指令；固定请求输出 64 tokens，temperature=0，忽略 EOS。词数不等于 token 数，实际用量保留在原始记录中。
- **联合 SLO**：同一个请求必须成功，并同时满足 **TTFT ≤ 1000 ms、TPOT ≤ 50 ms/token**。至少 **95% 的全部计划请求**达标，一轮才通过。
- **重复与时长**：每档重复 3 轮，每轮 8 次预热；正式请求数取 `max(300, ceil(req/s × 120))`，本次各档发送窗口约 120 秒。相邻重复轮反转档位顺序，降低运行顺序带来的偏差。
- **发压有效性**：要求全部请求发出、实际到达率误差不超过 5%、发送滞后 P95 不超过 100 ms。单请求总超时 30 秒，客户端最多允许 128 个未完成请求；触顶则记录未发送失败并判定负载无效，不静默降低到达率。
- **有效吞吐 goodput**：达标请求数除以从首个计划到达到最后完成或超时的总时长，**包含停止发送后的排空时间**。同时记录客户端 inflight 随时间变化、前后段均值及排空时长。

TTFT 从实际发送计时，到收到首个非空内容为止。TPOT 沿用现有客户端的 `(流式响应结束时间 - 首个内容时间) / (输出 tokens - 1)`，包含协议尾部开销，不代表逐 token 卡顿分位数。延迟分位数只统计成功请求，但失败和未发送请求仍计入联合达标率分母。客户端 inflight 也不等于 vLLM 内部等待队列。

默认 `gpu-memory-utilization=0.50`，为桌面应用保留显存；历史对照实验使用 0.85，因此两批数据不作为性能提升前后对照。结果增量写入 `results/capacity/<run-id>/`，包括逐请求 JSON、CSV、自动报告及服务配置和环境信息。详见[本次容量压测报告](results/capacity/20260907-capacity/report.md)。

只检查链路时可使用 `-Rates 2 -Repeats 1 -Requests 40 -MinDuration 0`，但短测结果不能用于判断持续容量。当前方案只给出固定配置、固定负载和有限运行窗口下的经验边界，不覆盖突发流量，也不承诺生产 SLA。

## 简历使用边界

完成真实实验后，可以描述为“基于 vLLM 搭建单卡推理服务”“构建异步流式压测工具”“评测 Continuous Batching 与 Prefix Cache”。不能描述为“实现 PagedAttention”“实现 Continuous Batching”或“证明端到端生产性能提升”。所有数字必须附带本机硬件、模型、并发、输入输出长度和对照配置。

## 三阶段 Prefix Cache 实验

[完整中文实测报告](results/prefix-reports/20260913/README.md) 包含 60 次运行、4,992 个测量请求、三阶段 CSV、6 份 PyTorch trace，以及可核验的原始数据。核心观察是 128→256 MiB 恢复前缀复用；two-hit 减少 94.1% 的 CPU L2 写入，但没有证明所有负载下端到端性能都更好。

正式数据使用统一的 `prefix_study` 入口复现，每次指定一个新的输出目录：

```powershell
python -m vllm_serving_lab.prefix_study --stage 1 --output results/prefix-study/my-stage1
python -m vllm_serving_lab.prefix_study --stage 2 --output results/prefix-study/my-stage2
python -m vllm_serving_lab.prefix_study --stage 3 --output results/prefix-study/my-stage3 --profile
python -m vllm_serving_lab.prefix_report --stage stage1 --input-dir results/prefix-study/my-stage1 --output results/prefix-reports/my-stage1.md
```

默认每组重复 2 次。第一阶段为原生缓存开关与 4 种会话场景；第二阶段固定 `gap0_bg2` 扫描 128/256/512 MiB；第三阶段固定 GPU 128 MiB、CPU 512 MiB，对比 `gpu-only`、`gpu-cpu-l2`、`gpu-cpu-l2-two-hit`。closed-loop 并发为 4，open-loop 测量恒定到达率，空闲间隔场景只在 closed-loop 下运行。以下 PowerShell 脚本保留作为手动探索入口；正式报告采用上面的统一入口及其缓存重置、计数核验和数据归档流程。

### 第一阶段：真实场景

`session-chat` 负载模拟客服或 Agent 的多轮会话。每个 session 有稳定的 system/tool 前缀，历史按轮次增长；`--idle-gap-ms` 控制会话空闲时间，`--background-unique-prefixes` 插入不同租户的冷前缀，制造 reuse distance 和 LRU 压力。请求记录包含 `session_id`、`prefix_id`、`expected_shared_tokens`、`reuse_distance`、`role`。

```powershell
.\scripts\Run-PrefixScenarios.ps1 -IdleGapsMs 0,30000 -BackgroundUniquePrefixes 0,4 -Requests 32 -Repeats 1 -Offline
```

脚本对每个场景分别重启 Prefix Cache 开启和关闭的服务，结果位于 `results/prefix-scenarios/<run-id>/`。`expected_shared_tokens` 是保留的旧字段名，内容为按英文单词数估计的共享长度，不是 tokenizer 结果或实际命中 token。正式报告的输入 token 来自 usage，命中 token 来自 vLLM Prometheus。历史是固定文本回放，不使用上一轮模型生成内容构造下一轮请求。

### 第二阶段：可变 GPU KV 容量

`--kv-cache-memory-bytes` 直接限制 vLLM 启动时的 GPU KV block pool。实验只改变容量，模型、并发、前缀流量和输出长度保持不变；每个容量都会重新启动独立进程。

```powershell
.\scripts\Run-KvCapacity.ps1 -CapacitiesMiB 128,256,512 -Requests 60 -Repeats 1 -Offline
```

重点观察容量降低后，不同 reuse distance 的 TTFT P95、成功率、goodput 和排队增长。这个参数是启动时的固定容量，不是运行时自动扩缩容。

### 第三阶段：GPU + CPU/Host L2

vLLM 0.10.2 V1 引擎提供 KV connector。仓库新增 `CpuL2Connector`，复用官方 `SharedStorageConnector` 的 tensor 读写路径，只增加 host tier 的容量上限和淘汰策略：`max_cpu_bytes` 超限按 LRU 删除完整 prefix 目录，`two_hit` 策略则要求同一前缀第二次出现后才进入 L2。

```powershell
.\scripts\Run-KvTiers.ps1 -GpuCapacityMiB 256 -CpuCapacityMiB 512 -Requests 60 -Offline
```

第三阶段比较 `gpu-only`、`gpu-cpu-l2` 和 `gpu-cpu-l2-two-hit`。L1 是 vLLM GPU Prefix Cache；正式实验的 L2 是容器内的 Linux tmpfs 内存文件系统，缓存每个 prompt 最前面的 512 个 token，按 16-token block 对齐，28 层全部写完后才发布为可命中条目。CPU/GPU 拷贝和 safetensors 序列化是同步路径。`two_hit` 按不同请求计数，第一次不写入，第二次开始允许写入；调度重试不能冒充第二次访问。这个实验 connector 不应描述成生产级 LMCache、异步 pinned-memory offload 或分布式缓存。

面试时可以按同一条主线复述：先用真实 reuse distance 建模流量，再固定 workload 扫 GPU 容量，最后增加一个有容量和淘汰策略的 host tier；每一步都用 TTFT P95、TPOT、goodput、SLO 和 profile 判断收益来自缓存命中、缓存搬运还是排队。
