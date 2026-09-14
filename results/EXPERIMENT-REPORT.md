# vLLM Serving Lab 实验数据总报告

更新时间：2026-09-15。本文把项目中已经完成的服务、Prefix Cache、容量、SLO、多级 KV Cache、租户隔离和 profile 实验放在同一个数据入口中。表格使用各次运行的中位数；每次运行的原始请求记录仍保留在对应目录，不能把不同实验的中位数拼成一条新的样本分布。

## 1. 环境与数据范围

| 项目 | 固定值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU，8,188 MiB |
| 模型 | Qwen2.5-1.5B-Instruct，FP16 |
| vLLM | `vllm/vllm-openai:v0.10.2`，V1 engine |
| 上下文上限 | 2,048 token |
| 输出设置 | 32 token（SLO 容量实验为 64 token），temperature=0，ignore EOS |
| 服务端并发 | 历史基线为 `max-num-seqs=8`；三阶段和租户实验为 `max-num-seqs=4` |
| Prefix Cache | 对照实验显式开关；三阶段和租户实验开启 |
| 测量 | 客户端单调时钟 TTFT/TPOT/P50/P95，Prometheus counter/histogram，必要时 NVIDIA SMI 和 PyTorch profiler |

正式数据按实验批次统计如下：

| 实验批次 | 运行数 | 测量请求 | 数据位置 |
|---|---:|---:|---|
| Continuous Batching / Prefix Cache 历史基线 | 18 | 1,080 | `results/report.md`、`results/raw/20260818-rtx4070-fp16/` |
| SLO 开环容量扫描 | 9 | 5,400 | `results/capacity/20260907-capacity/report.md` |
| 三阶段 Prefix/KV Cache | 60 | 4,992 | `results/prefix-reports/20260913/`、`results/prefix-study/published-20260913/` |
| 全局共享 vs 租户命名空间 | 8 | 768 | `results/prefix-reports/20260915/tenant.md`、`results/prefix-study/tenant-namespace-20260915/` |
| 合计（不同实验批次相加） | 95 | 12,240 | 本文汇总，不把不同批次合并计算 |

正式实验都在模型加载、编译和预热完成后开始；三阶段和租户实验在每次 run 前重置 GPU Prefix Cache、CPU L2 内容和 two-hit admission history。open-loop 记录实际到达率、dispatch lag、inflight 和 drain time，closed-loop 记录完成反馈下的吞吐。SLO goodput 包含最后一个请求的排空时间。

## 2. Continuous Batching 与服务参数实验

### 2.1 测试目标

项目没有把 vLLM 的 scheduler 删除来“实现” Continuous Batching，而是用 `max-num-seqs=1` 作为近似串行控制，用 `max-num-seqs=8` 观察同一个 vLLM 服务允许多个 active sequence 时的连续调度效果。Paged KV Cache 是 vLLM engine 的基础机制，没有干净的运行时 on/off 开关；混合长度 prompt 用于验证它在变长请求下工作，不宣称一个孤立的 PagedAttention 加速百分比。

### 2.2 实测矩阵与数据

| 配置 | 客户端并发 | `max-num-seqs` | Prefix Cache | TTFT P50 ms | TTFT P95 ms | E2E P95 ms | 输出 tok/s | 请求 req/s |
|---|---:|---:|:---:|---:|---:|---:|---:|---:|
| `cb_serial` | 8 | 1 | off | 46,824.82 | 55,439.06 | 62,087.94 | 9.28 | 0.15 |
| `cb_batch` | 8 | 8 | off | 316.60 | 512.60 | 2,212.22 | 259.05 | 4.05 |
| `cb_scale_1` | 1 | 8 | off | 408.53 | 1,187.22 | 6,707.54 | 10.87 | 0.17 |
| `cb_scale_4` | 4 | 8 | off | 4,845.86 | 16,418.17 | 33,075.89 | 16.80 | 0.26 |

同一客户端并发 8 下，`max-num-seqs=1 -> 8` 的输出吞吐为 `9.28 -> 259.05 tok/s`，提升约 26.9 倍；E2E P95 为 `62.09 -> 2.21 s`。这说明 active sequence 上限是这个负载的强瓶颈，但不能把结果表述成“单独实现了 Continuous Batching”。

### 2.3 影响结果的超参数

| 参数 | 本项目作用 | 本次对照 |
|---|---|---|
| 客户端 concurrency | 同时在飞的请求数；决定服务是否有机会组成 batch | 1、4、8 |
| `max-num-seqs` | scheduler 同时保留的 active sequence 上限；过小会串行化，过大可能增加显存和调度开销 | 1、8 |
| `max-num-batched-tokens` | 每个调度步可处理的 token budget；过小会增加 prefill 分片，过大可能挤压 decode | vLLM 镜像默认值，三阶段日志为 8192 |
| `enable-chunked-prefill` | 是否把长 prefill 分成调度块，影响 prefill/decode 互相阻塞 | 固定为镜像默认，未作为独立变量 |
| `max-model-len` | KV pool 的最大上下文规划，影响可分配 block 数和长请求可接受性 | 2048，固定 |
| `gpu-memory-utilization` / `kv-cache-memory-bytes` | 决定模型、workspace 和 KV pool 的显存预算 | 历史基线 0.85；三阶段直接扫描 KV bytes |
| 输出长度、temperature、EOS | 改变 decode 工作量和 TPOT | 固定输出长度，temperature=0，ignore EOS |
| Prefix Cache | 命中时减少 prefill；会改变 scheduler 负载而非只改变一个计数 | 历史 `pc_off/pc_on`；三阶段开启 |

因此面试时可以把这个部分概括为：先固定模型和输入，只改变 active sequence 上限与客户端并发，观察吞吐、排队和尾延迟；然后把 `max-num-batched-tokens`、chunked prefill、显存预算列为必须控制的混杂变量。

## 3. Prefix Cache 开启与关闭

### 3.1 历史共享前缀实验

该实验使用约 960 个英文词的稳定共享前缀、8 路客户端并发、`max-num-seqs=8`。除 Prefix Cache 外其他参数完全相同，每组 3 次运行。

| 配置 | TTFT P50 ms | TTFT P95 ms | E2E P95 ms | 有效输入 tok/s | 输出 tok/s | 请求 req/s |
|---|---:|---:|---:|---:|---:|---:|
| `pc_off` | 8,753.23 | 31,243.99 | 57,947.33 | 211.89 | 6.64 | 0.21 |
| `pc_on` | 204.93 | 1,099.21 | 3,326.53 | 4,220.85 | 132.37 | 4.14 |

TTFT P50 下降约 97.66%，有效输入吞吐提高约 18.9 倍。这个收益只适用于 token 前缀真的相同且命中率足够高的流量；第一轮冷请求仍需要完整 prefill。

### 3.2 长时间不对话的解释边界

vLLM 的 Prefix Cache 是有限 GPU KV block pool 中的可复用块。它没有“只因为用户一段时间没说话就必然保留”的 TTL 保证；只要 block 被容量压力、调度需要或显式 reset 淘汰，下一次对话就会重新 prefill。第一阶段只测了 0 和 2,000 ms 的 idle gap，不能推出小时级保留时间。是否丢失取决于容量、reuse distance、其他租户流量和淘汰策略。

## 4. SLO 开环容量实验

### 4.1 口径

固定一个服务配置：Prefix Cache 关闭，`max-num-seqs=8`，`gpu-memory-utilization=0.50`，输入实际为 154/538/1050 token 三种长度，输出 64 token。客户端按均匀外部到达率发送，连续约 120 秒，每个速率重复 3 次。

联合 SLO 为：请求成功、`TTFT <= 1000 ms` 且 `TPOT <= 50 ms/token`；每次至少 95% 的全部计划请求达标，同时实际到达率误差不超过 5%、dispatch-lag P95 不超过 100 ms。Goodput 为达标请求数除以包含 drain 的总时间。

### 4.2 数据

| 目标到达率 | 通过重复数 | 联合 SLO 范围 | TTFT P95 中位数 ms | TPOT P95 中位数 ms/token | Goodput req/s | Drain 中位数 s |
|---:|---:|---:|---:|---:|---:|---:|
| 4 req/s | 3/3 | 100% | 123.77 | 25.25 | 3.969 | 1.20 |
| 5 req/s | 3/3 | 100% | 335.97 | 26.24 | 4.958 | 1.23 |
| 6 req/s | 0/3 | 3.33%--4.58% | 20,376.50 | 26.24 | 0.231 | 22.57 |

这组实验给出的本机、这组 prompt 和这组参数下的 sampled boundary 是 5 req/s；6 req/s 连续三次失败。6 req/s 的主要失败来自 TTFT 排队，TPOT 仍低于阈值；输出 tok/s 只从 317.33 增至 323.60，但 inflight 和 drain 明显积累，所以“请求最后都成功”不能等同于服务容量足够。4--5 req/s 之间没有细扫，不能声称精确阈值。

## 5. 第一阶段：真实场景 Prefix Cache 评测

### 5.1 场景设计

`session-chat` 模拟客服/Agent 多轮会话：4 个 session、每个 8 轮，固定 system/tool 前缀，历史按固定文本追加。`bg2` 在每个前台请求前插入 2 个一次性冷请求；`gap2000` 在同一 session 返回时至少等待 2 秒。每个场景分别测 Prefix Cache off/on，closed-loop 并发 4；无 idle gap 的场景额外以 6 req/s 做 open-loop。

### 5.2 Closed-loop 数据

| 场景 | 含义 | 续聊 TTFT P50：off/on ms | 整体 TTFT P95：off/on ms | 吞吐：off/on req/s | on 的 GPU 命中 token |
|---|---|---:|---:|---:|---:|
| `gap0_bg0` | 立即续聊、无冷背景 | 345.1 / 42.6 | 1,082.5 / 455.7 | 3.74 / 6.45 | 83.4% |
| `gap0_bg2` | 立即续聊、每轮 2 个冷背景 | 249.1 / 124.8 | 600.4 / 186.4 | 4.20 / 5.95 | 37.0% |
| `gap2000_bg0` | 间隔 2 秒、无冷背景 | 590.4 / 145.6 | 813.4 / 479.4 | 1.33 / 1.60 | 83.4% |
| `gap2000_bg2` | 间隔 2 秒、带冷背景 | 133.7 / 37.0 | 224.7 / 179.9 | 4.28 / 4.70 | 37.0% |

固定到后续阶段的 `gap0_bg2` 在 open-loop 6 req/s 下，off/on 的 TTFT P95 为 `3455.0/184.0 ms`，SLO 达标率为 `6.25%/100%`，goodput 约为 `0.30/5.87 req/s`。冷请求会降低全局命中率，但也会改变分母；不能只看命中率下降就断言发生了同等比例的淘汰。

## 6. 第二阶段：可变 GPU KV Cache 容量

固定 `gap0_bg2`，只重启服务并改变 `--kv-cache-memory-bytes`；模型、prompt 顺序、并发、输出和负载保持一致。容量是启动时固定的敏感性扫描，不是运行中自动扩缩容。每个容量做 closed-loop 并发 4、open-loop 3/6 req/s，各 2 次。

### 6.1 端到端数据

| GPU KV 上限 | 实际 token pool | Closed GPU 命中率 | Closed 续聊 TTFT P50 ms | Closed TTFT P95 ms | Closed req/s | Open-3 TTFT P95 ms | Open-6 TTFT P95 ms | Open-6 goodput req/s | Open-6 SLO |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 MiB | 4,672 | 1.47% | 195.7 | 257.2 | 5.49 | 99.2 | 863.9 | 1.64 | 29.17% |
| 256 MiB | 9,360 | 36.93% | 129.5 | 180.0 | 6.04 | 69.9 | 321.3 | 5.40 | 92.19% |
| 512 MiB | 18,720 | 36.97% | 129.8 | 180.4 | 6.03 | 69.2 | 221.0 | 5.71 | 97.40% |

128 MiB 已不足以保留这个 reuse distance 下的热前缀；256 MiB 后命中率和 closed-loop 吞吐基本恢复，256->512 MiB 的收益很小。可以说拐点在已采样的 128 和 256 MiB 之间，不能把 256 MiB 宣称成精确最优容量。6 req/s 的 512 MiB 两次重复并非都超过 95%，所以也不能把它包装成生产容量保证。

## 7. 第三阶段：多级 KV Cache

### 7.1 实现和实验组

固定 GPU KV 为 128 MiB、CPU L2 为 512 MiB，三组使用同一 `gap0_bg2` 请求序列：

| 组别 | GPU miss 后的路径 | CPU admission |
|---|---|---|
| `gpu-only` | 重新 prefill | 无 L2 |
| `gpu-cpu-l2` | CPU 命中则 H2D 回载，否则 prefill | 首次访问即可写入，容量超限 LRU |
| `gpu-cpu-l2-two-hit` | CPU 命中则 H2D 回载，否则 prefill | 同一 prefix 第二个不同请求出现后才准入，容量超限 LRU |

connector 只保存 prompt 最前面的 512 token，按 16-token block 对齐，28 层全部写完并发布完成标记后才可读。L2 使用容器内 Linux tmpfs；tensor D2H/H2D、safetensors 序列化、索引和同步等待都位于请求关键路径。这是用于测量 trade-off 的单机实验 connector，不是异步 pinned-memory、分布式 LMCache 或生产级远端 KV 服务。

### 7.2 端到端数据

| 组别 | Closed 续聊 TTFT P50 ms | Closed TTFT P95 ms | Closed req/s | Open-6 TTFT P95 ms | Open-6 goodput req/s | Open-6 SLO |
|---|---:|---:|---:|---:|---:|---:|
| GPU-only | 195.9 | 256.5 | 5.52 | 689.7 | 3.05 | 54.17% |
| GPU + CPU L2 LRU | 224.2 | 273.8 | 5.23 | 1,587.9 | 0.50 | 9.38% |
| GPU + CPU L2 two-hit | 170.5 | 243.5 | 5.62 | 513.2 | 2.85 | 49.48% |

### 7.3 LRU 与 two-hit 的成本解释

普通 CPU L2 LRU 在这组请求中发生 68 次 store、32 次 eviction，写入约 952.16 MiB；每个新前缀第一次访问就执行同步 D2H、safetensors 序列化、tmpfs 文件操作和 CUDA 同步。closed-loop 中上一个请求结束后才发下一个请求，因此保存成本直接压低下一次请求的启动速度，吞吐由 5.52 降到 5.23 req/s；open-loop 6 req/s 中这些同步保存又会占用 worker 时间并积累请求排队，TTFT P95 由 689.7 升到 1,587.9 ms，goodput 由 3.05 降到 0.50 req/s。

two-hit 把写入从 68 次降到 4 次，写入量从 952.16 降到 56.01 MiB，减少约 94.1%；代价是仍有 24 次 CPU load、读出 336.06 MiB。closed-loop 里它的写放大显著下降，吞吐略高于普通 LRU；open-loop 6 req/s 仍要同步 H2D、safetensors 读取和索引，24 次回载被外部到达率放大，goodput 2.85 req/s 仍略低于 GPU-only 的 3.05 req/s。也就是说 two-hit 的确定收益是减少一次性流量污染和写成本，不能表述成所有负载下都会端到端加速。

### 7.4 Profile 与正确性

独立的三次相同 prompt、每次先清空 GPU Prefix Cache 的诊断得到：

| 组别 | CUDA kernel 事件数 | kernel duration 总和 ms | H2D MiB | D2H MiB |
|---|---:|---:|---:|---:|
| GPU-only | 18,486 | 777.1 | 0.075 | <0.001 |
| CPU L2 LRU | 18,626 | 992.0 | 28.384 | 14.000 |
| CPU L2 two-hit | 18,570 | 932.4 | 14.284 | 14.000 |

普通 L2 的第 1 次写入、后 2 次回载；two-hit 的第 1 次拒绝、第 2 次写入、第 3 次回载。两种 L2 与冷启动重算得到的 greedy output 相同；profile 只用于证明真实 memcpy/算子路径，不把 trace 中重叠的 CUDA stream 时间相加成请求延迟。

## 8. 全局共享 vs 共享物理池 + 租户命名空间

### 8.1 场景和安全语义

4 个租户各 8 轮对话；公共 policy/tool 前缀完全相同，私有 workspace 和追加历史不同；每个前台请求前插入 2 个一次性冷请求，共 96 请求/run。两组都使用 GPU 128 MiB、CPU L2 512 MiB、two-hit、closed-loop 并发 4 和 open-loop 6 req/s。prompt 字节、请求顺序、模型和容量完全一致，唯一变量是 namespace 元数据。

`global-shared` 不发送 `cache_salt`，因此相同公共前缀可以跨租户复用。`tenant-namespaced` 给每个前台租户发送稳定的 vLLM `cache_salt`；冷背景请求使用各自一次性 salt，CPU L2 的 key 也包含相同 namespace。这里是“一个物理池、逻辑 key 隔离”，不是每个租户分配独立 CPU 内存，也不是加密和鉴权机制；生产系统仍需在 API 层做身份校验和数据保护。

### 8.2 端到端数据

| 组别 | Load | GPU 命中 token | TTFT P50 ms | TTFT P95 ms | E2E P95 ms | req/s | Goodput req/s | SLO |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| global-shared | closed | 28.46% | 133.26 | 204.75 | 846.52 | 4.87 | 4.72 | 96.88% |
| global-shared | open-6 | 28.46% | 1,341.45 | 2,739.94 | 3,439.63 | 4.99 | 0.34 | 6.77% |
| tenant-namespaced | closed | 0.00% | 215.41 | 305.26 | 925.08 | 4.66 | 4.45 | 95.31% |
| tenant-namespaced | open-6 | 0.00% | 1,896.19 | 3,506.68 | 4,236.60 | 4.78 | 0.32 | 6.77% |

### 8.3 L2 证据与 trade-off

| 组别 | CPU loads | CPU stores | admission rejects | 写入 MiB | 读出 MiB | Resident MiB |
|---|---:|---:|---:|---:|---:|---:|
| global-shared | 0 | 1 | 65 | 14.00 | 0.00 | 14.00 |
| tenant-namespaced | 24 | 4 | 68 | 56.01 | 336.06 | 56.01 |

全局共享组的公共前缀可以留在 128 MiB GPU tier，因此没有发生 CPU load；命名空间组虽然共享同一物理 CPU pool，却不能跨租户拿 GPU 前缀，24 次 load 和 336 MiB read 直接体现了隔离成本。命名空间让同一文本在不同租户之间不复用，降低了跨租户数据泄漏风险和 side-channel 风险，但牺牲了命中率、TTFT 和 open-loop goodput。是否采用它应由数据隔离要求和 prefix 重合度共同决定，而不是简单追求一个全局 hit ratio。

本组审计通过：8/8 run、768/768 请求成功；prompt hash 跨策略一致；实际到达率有效；CPU resident 未超过 512 MiB；全局组观察到 0 load、命名空间组观察到 24 load 且每次 load 都命中 L2。这也是为什么报告没有强行要求两种策略都必须产生 CPU load：全局组的“零 load”本身就是 GPU 共享收益的结果。

## 9. 其他已做实验和校验

1. **SLO pilot**：曾用很短的发送窗口试探 4/6 req/s；短窗口在队列积累前可能看起来达标，已单独保留在 `results/capacity/20260907-pilot/`，没有混入正式 120 秒容量结论。
2. **L2 correctness/admission**：验证完整 28 层快照写完才可读、GPU miss 后实际 load、LRU 容量淘汰、two-hit 首次访问拒绝、同一请求重试不伪造第二次访问、reset 清理 admission history，以及不同策略的 greedy output 一致。
3. **Prometheus/系统采样**：记录 `prefix_cache_queries_total`、`prefix_cache_hits_total`、request queue/prefill/decode histogram、waiting/active KV gauge，并在可用时采样 GPU util、显存和温度。WDDM 下异常显存值会过滤并保留计数；GPU gauge 不能等同于所有可复用前缀占用。
4. **旧 smoke/partial runs**：`actual-stage1-smoke`、`actual-stage2`、`stage1-partial` 和早期 `smoke-l2-run1.json` 用于排查链路或旧 workload，不参与本文正式表格。正式三阶段使用修正后的稳定前缀、冷请求、cache reset、counter 和 prompt hash 审计。

## 10. 复现入口

```powershell
# 历史基础矩阵
python -m vllm_serving_lab.benchmark --help
python -m vllm_serving_lab.summarize --help

# 三阶段正式实验（每次使用新目录）
python -m vllm_serving_lab.prefix_study --stage 1 --output results/prefix-study/replay-stage1
python -m vllm_serving_lab.prefix_study --stage 2 --output results/prefix-study/replay-stage2
python -m vllm_serving_lab.prefix_study --stage 3 --output results/prefix-study/replay-stage3 --profile

# 租户 namespace 对照
python -m vllm_serving_lab.tenant_study --output results/prefix-study/replay-tenant
python -m vllm_serving_lab.prefix_report --stage tenant `
  --input-dir results/prefix-study/replay-tenant `
  --output results/prefix-reports/replay-tenant.md
```

三阶段报告生成器会检查完整矩阵、客户端与 Prometheus counter、open-loop 到达率、跨配置 prompt hash、CPU 容量和 L2 correctness。所有测量都依赖 Linux Docker engine、NVIDIA GPU 和本地模型缓存；换机器后应把结果视为新实验，不能直接复用这里的 latency 数字。

## 11. 面试复述主线

可以按“流量局部性 -> GPU 容量 -> host tier 成本 -> 多租户隔离”复述：先用稳定前缀、追加历史、idle gap 和冷背景构造接近真实的会话流量；再固定 workload 扫 GPU KV 容量，证明 128 MiB 的重算和 256 MiB 左右的拐点；之后接入有限 CPU L2，比较普通 LRU 与 two-hit admission，用 loads/stores/bytes 和 PyTorch profile 拆开“重算、搬运、序列化、排队”；最后在相同物理池上加入 vLLM `cache_salt` 和同 namespace 的 CPU key，量化全局共享效率与租户隔离开销。

需要明确的边界是：Continuous Batching、Paged KV Cache、原生 Prefix Cache 和 scheduler 来自 vLLM；本项目实现的是 workload、可复现实验编排、容量受限的 CPU L2 connector、LRU/two-hit admission、租户 namespace 对照和数据审计。这里的同步 tmpfs L2 证明了 trade-off 和测量方法，不等于生产级异步 offload、分布式 KV 服务或普适的端到端加速。
