# vLLM CPU L2：LRU / LFU / ARC 吞吐与开环容量实验方案

日期：2026-10-08（Asia/Shanghai）。状态：2026-10-08 设计基线；执行状态与结果以批次记录和审计报告为准。新增实验拟作为 vllm-serving-lab 的“CPU L2 替换策略”批次，完成后录入实验总报告。

## 1. 研究问题与范围

比较 CPU L2 的 LRU、LFU、ARC 缓存替换策略，回答：

1. CPU 缓存容量低/高时，哪种策略提高吞吐、Goodput 和已测最高达标到达率？
2. 长、短输入下，缓存复用节省的 prefill 是否足以覆盖保存/回载和策略维护成本？
3. 稳定热点、一次性扫描、热点切换、无复用流量下，策略的优势是否改变？
4. 长输出增加 decode 压力后，替换策略收益是否被稀释？

本方案中的三种策略作用于 **CPU L2 的替换/淘汰**。vLLM 的 GPU KV 管理、请求调度器、batch token budget 和数据搬运路径在三组中保持相同。

现有 Connector 固定缓存前 512 token，约 14 MiB/完整前缀；实现入口为 `projects/vllm-serving-lab/src/vllm_serving_lab/cpu_l2_connector.py`。主实验保留此对象粒度，用输入后缀长度改变请求计算量。这样三种策略管理的对象等大，可以按原始 ARC 的单位容量定义比较。ARC 原论文假设 uniform page sizes，不能直接把不同大小的完整 prompt 当作等价 entry：

- 原论文：https://www.usenix.org/legacy/event/fast03/tech/full_papers/megiddo/megiddo.pdf

## 2. 实验组与统一语义

| 组别 | CPU L2 行为 | 参数与约束 |
|---|---|---|
| G0：GPU-only | GPU miss 后重算，无 CPU L2 | 每个场景/请求类型一份对照；不按 CPU 容量重复 |
| G1：LRU | 淘汰最久未访问的 resident prefix | 使用统一策略接口实现，避免沿用目录扫描而其他策略走内存索引造成实现路径混杂 |
| G2：LFU | 淘汰访问次数最少的 resident prefix | 入驻频次 1；一次有效 CPU 访问增加 1；并列时按 LRU；淘汰后频次清零；主实验不加 aging |
| G3：ARC | T1/T2 resident + B1/B2 ghost，自适应调整 p | 使用经典 ARC 状态转换；ghost 仅保存 key；按等大对象的 slot 数管理 |

统一准入：外部缓存未命中且完成该前缀计算时允许保存；所有三组关闭现有 two-hit 准入。ARC 的替换行为不等同于 two-hit 写入过滤，不能预设 ARC 会减少首次冷写入。若以后研究 two-hit，则对三种策略都加相同准入规则，另立消融组。

统一访问语义：GPU 已经覆盖完整 512-token prefix 时不触碰 CPU 替换状态；确实需要外部 prefix 的请求，按 request_id + prefix_key 去重，一次逻辑访问至多更新一次频次/recency/ARC 状态。记录 scheduler lookup、实际 load、实际 store 三类事件，不能把多层加载、重试或多次调度当作多个业务访问。

缓存状态由单一权威状态机维护；scheduler 与 worker 的消息带访问序号、generation 和完成状态，避免两边独立维护不一致的 LFU/ARC。正在回载、保存或被请求引用的内容不能被提前删除；每次淘汰需明确 victim 和理由。

执行实现：使用客户端唯一请求 ID 和 generation 标识逻辑访问；worker 同步完成完整层快照后发布 complete，替换前为该 worker step 中的 load/store 对象提供保护。ARC 的 p 更新保留原论文比值，不做整数截断。额外状态对象大小按 CPython getsizeof 去重计量，框架、allocator、文件缓存和统一 staging 不归入 CPU 内容预算。

## 3. 固定变量

| 变量 | 固定值 |
|---|---|
| 硬件 | 当前 RTX 4070 Laptop；同一供电/功耗模式；独占实验 GPU |
| 模型/框架 | Qwen2.5-1.5B-Instruct FP16；vLLM 0.10.2 V1；记录模型 revision、镜像 digest、代码 commit |
| 上下文 | 2048 token |
| GPU KV | 256 MiB，所有策略/CPU 容量/主场景保持一致 |
| 服务端并发 | max-num-seqs=4 |
| batch token budget | 8192；优先显式关闭 chunked prefill，检查实际日志；本组输入在并发 4 下不构造跨步 prefill |
| Prefix Cache | 原生 GPU Prefix Cache 开启；命名空间规则一致 |
| CPU 数据路径 | Linux RAM-backed tmpfs + safetensors；同样的同步 D2H/H2D、序列化和完整层发布协议 |
| 缓存对象 | 每个 prefix 前 512 token，16-token block 对齐，28 层完整快照 |
| 输出/采样 | 主实验 32 token；temperature=0；ignore EOS |
| 编译预热 | 在正式运行前完成；随后清理 GPU Prefix Cache、CPU 内容、LFU 频次、ARC ghost/p 和统计 |
| 配对负载 | 同一场景/类型/速率/repeat 的所有策略使用相同 prompt token IDs、顺序、到达计划与 seed |

选择 GPU 256 MiB，是因为既有日志对应约 9360-token pool，能覆盖并发 4 的长输入活动集，同时 32 个 512-token 热前缀总计 16384 token，超过 GPU pool。实际 pool 必须在新启动日志复核。不得按不同 CPU 策略分别调整 GPU 容量。

每个 prefix 的首个 16-token block 即区分 key，避免不同 key 意外共享大段 GPU prefix。相同 key 的最前 512 token 完全一致，后缀随请求改变。长度通过模型 tokenizer 和实际 usage 校验，使用 token 数，不使用单词数；必要时扩展客户端以接受 token-ID prompt。

## 4. 自变量：容量、请求类型、访问模式

### 4.1 CPU 容量

| 档位 | KV 内容预算 | 预期完整对象数 | 与热点工作集的关系 |
|---|---:|---:|---|
| C-low | 128 MiB | 约 9 个 | 32 个热点前缀明显放不下 |
| C-high | 512 MiB | 约 36 个 | 可容纳 32 个热点前缀及少量冷对象 |

正式运行前测量单对象实际字节 B，公布 c=floor(C/B)。512-token payload 理论约 14 MiB，safetensors header 也计入内容预算。ready + reserved 内容必须不超过 C；写入前预留容量并安全淘汰，记录写入中的临时占用，不能只统计淘汰后的 resident。metadata、ARC ghost 和统一 staging buffer 另列实际内存开销；不能称 C 为进程总 RAM 上限。

中间容量 256 MiB 保留为后续拐点扩展，不放入首轮完整矩阵。容量结论同时标记 C/当前热点工作集大小。

### 4.2 请求长度

| 类型 | 总输入 | 可缓存 prefix | 输入后缀 | 输出 | 用途 |
|---|---:|---:|---:|---:|---|
| P-short | 640 token | 512 | 128 | 32 | 短输入、短输出，观察回载固定成本 |
| P-long | 1664 token | 512 | 1152 | 32 | 长输入、短输出，观察额外 prefill 和排队压力 |

这里改变的是总输入长度，缓存 prefix 仍为 512 token，因此 **不把长输入实验表述为更长缓存对象的回载收益**。不同请求类型的绝对 req/s 不能直接用来排名策略；策略比较在同一类型内部进行。

### 4.3 四类访问场景

| 场景 | 固定生成规则 | 工作集/变化 | 主要验证点 |
|---|---|---|---|
| S1：稳定热点 | 80% 请求访问 32 个固定热 key，热访问按 Zipf(s=1.1)；20% 为首次且唯一的冷 key | 热点集合与分布全程固定 | 稳定频次是否帮助 LFU，容量高低是否改变优势 |
| S2：扫描污染 | 重复 320 请求周期：128 热访问（32 key × 4 轮）→64 全新冷 key→128 返回同一热集合；每周期冷 key 不重复 | 一次扫描约 896 MiB，超过两个 CPU 档位 | 热前缀是否被扫描挤出；扫描后命中和 TTFT 恢复速度 |
| S3：热点切换 | 三个等长阶段 A→B→C，各有互不重叠的 32 个热 key；各阶段 80% 热、20% 唯一冷；热分布均为 Zipf(s=1.1) | 两次热点切换；闭环按请求序号三等分，开环按预定发送窗口三等分 | LFU 旧频次滞留、LRU 跟踪新近访问、ARC 自适应过程 |
| S4：无复用 | 100% 唯一 prefix key；不再出现第二次 | 没有可利用的前缀复用 | 负对照：策略、写入和搬运的纯开销，以及 GPU-only 是否更好 |

Zipf、比例、工作集规模在预实验后冻结；不得按某个策略的表现重新挑选 trace。S2 与 S3 要分别输出扫描/切换前后时间窗结果，全程均值不能掩盖恢复速度。预期优势只是假设，不预设 ARC 获胜。

## 5. 主矩阵与扩展矩阵

主矩阵：3 CPU 策略 × 2 CPU 容量 × 2 请求类型 × 4 场景 = **48 个 CPU 配置**。

GPU-only：2 请求类型 × 4 场景 = **8 个配置**。主实验合计 **56 个配置**。每个配置都测闭环吞吐，并搜索自己的开环最高达标速率。

长输出扩展仅在 S1、CPU 128 MiB 下追加：P-short/P-long 的输出由 32 改为 256 token，三种 CPU 策略共 6 配置，加 2 个 GPU-only，共 **8 配置**。与主矩阵相同条件的 32-token 输出组配对，覆盖短输入长输出和长输入长输出；最大总长 1664+256=1920，仍在 2048 内。

混合长短请求保留为第二批扩展：50% P-short + 50% P-long，同一热 key 可出现在两种输入长度中，输出仍为 32。首轮先使用纯短/纯长组识别主效应，避免把长度比例与访问局部性一起改变。

## 6. 测量协议

### 6.1 预实验与机制检查

先跑 S1/S3 × 两档 CPU × P-short × 三种策略 = 12 CPU 配置，加 2 GPU-only，共 **14 个预实验配置**；另做少量 P-long 和 256-token 输出契约检查。预实验使用短 trace，只检查可运行性、长度、实际 CPU load、淘汰、ghost 更新和容量预算，不作为容量证据。

重点检查 GPU 命中是否遮蔽 CPU 策略：必须观察到 CPU lookup/load 和低容量下 eviction。预先设机制门槛：S1/S2/S3 的指定诊断 trace 中，CPU lookup 请求占比至少 20%；每个有 CPU eviction 目标的配置都要有 eviction 证据。若不满足，统一修改所有策略共享的热 key 数量或 GPU KV 预算后重新预实验，冻结最终方案，不保留有利的旧结果。

另做离线策略 trace：容量 4/9/36 slots 下逐项核对 LRU/LFU victim 和 ARC T1/T2/B1/B2/p、不变量、去重/reset。它验证算法实现，不替代 GPU 服务实验。相同强制 GPU miss prompt 的冷重算与 CPU 回载 greedy 输出必须一致。

### 6.2 闭环吞吐

- concurrency=4；每轮固定 **960 个请求**；S2 正好覆盖三个扫描周期，S3 每阶段 320 个请求。
- 三个独立 seed，每个策略使用配对 trace；每轮完成全部请求并包含初始冷缓存阶段。
- 三种 CPU 策略按 LRU→LFU→ARC、LFU→ARC→LRU、ARC→LRU→LFU 轮换顺序；GPU-only 的运行位置也轮换。
- 报告 req/s、output tok/s、Goodput、TTFT/TPOT/E2E P50/P95、每轮值和中位数。
- 若存在 invalid run 或模型/驱动错误，保留原始记录并重跑配对组；不能只补最快的策略。

主矩阵闭环共 **168 轮、161280 个计划请求**。首次 960 请求不是“提前将缓存填满的隐藏预热”；正式冷启动到稳定阶段均保留。

### 6.3 开环容量：最高已测达标请求速率

统一外部均匀到达，不等待完成，不用客户端 semaphore 悄悄降低目标速率。每个容量点都从清空的 GPU/CPU/策略状态开始。

**主联合 SLO：请求成功 AND TTFT≤1000 ms AND TPOT≤50 ms/token；每轮至少 95% 的全部计划请求同时达标。** 三阶段既有的 TTFT≤300 ms / TPOT≤40 ms/token 作为次级更严格口径，分别输出，不混用阈值。

发压有效性：全部计划请求真实派发，实际到达率误差≤5%，dispatch-lag P95≤100 ms。超时、失败、未派发请求仍计入达标率分母；客户端 safety cap 固定为 512，达到上限时标记发压无效，不声称服务容量失败。

搜索步骤：

1. 探索：60 秒/点，1 次；从 1、2、4、8 req/s 找通过/失败候选；必要时向下到 0.5/0.25，向上到 12/16/24/32。只用于找范围。
2. 细扫：在候选区间二分，直到速率分辨率≤0.25 req/s；若观察到非单调结果，保留相邻点并扩展局部网格，不强行二分推断。
3. 正式确认：候选最高通过点及相邻较高失败点，各 **至少 300 秒、至少 300 请求、3 次独立 seed**。以正式长窗口为准，探索点通过不能算最高容量。
4. 若确认失败则向下找并重新确认；若较高点三次都通过则继续向上。若上点是混合通过/失败，则追加 seed 404/505 至 5 次并标记不稳定区间，不能丢弃失败轮。没有稳定失败上点时只报告已测下界。

完整发送窗口使用 n=max(300,ceil(300×rate)+1)，因为 n 个等间隔请求的发送跨度是 (n−1)/rate；每轮保留实际跨度。共同速率对照采用 120 秒、至少 60 请求、seed 909，不作为容量边界确认。

额外有限窗口积压判据：正式轮最后 60 秒的客户端 inflight 均值不得比前一个 60 秒高出 max(3 个请求，前窗均值的 10%)。这是预先声明的工程判据；同时保留趋势图、内部 waiting 指标与 drain，不把客户端 inflight 称为内部队列。主 SLO 通过但积压判据失败的点单独标记，不作为“稳定通过点”。

最高已测达标速率：所有正式重复都满足联合 SLO、发压有效性和积压判据的最高点 λ_pass。输出 λ_pass、相邻不稳定/稳定失败点、确认窗口和重复数；不把有限窗口结果称为精确生产容量。对于全部低速点仍有固有延迟违约的场景，报告“当前阈值下未找到通过点”，不临时放宽 SLO。

为可比性，容量搜索后另列各场景/类型所有策略共同测试过的绝对到达率结果；至少补一个不高于所有有效 λ_pass 的共同点。各自 λ_pass 的延迟和 Goodput 表不当作等负载对照。

## 7. 指标与解释

| 层次 | 必须记录 |
|---|---|
| 用户体验/服务 | 请求成功率、全部计划请求联合 SLO%、Goodput（包含完整 drain）、req/s、output tok/s、TTFT/TPOT/E2E P50/P95 |
| 发压/队列 | scheduled/dispatched/completed 时间、实际 rate、dispatch lag、client inflight、server waiting、preemptions、drain |
| GPU | 原生 prefix 查询/命中 token、实际 KV pool、活动 KV、GPU util/功耗/温度 |
| CPU L2 | lookup/hit/load/store/eviction、实际重用 token、读写 MiB、ready/reserved/peak bytes、save/load wall time |
| 策略 | 每次更新/淘汰成本，LFU frequency 分布，ARC p/T1/T2/B1/B2，metadata bytes，victim reason |
| 场景 | 请求类型、prefix key、hot/cold/scan/phase 标记、reuse distance、prompt/token hash |

CPU 条件命中率=实际 CPU 命中/有效 CPU lookup；全局外部复用比例=CPU 重用 token/所有输入 token。与 GPU 命中率分开列，避免把 GPU 已命中的请求放进 CPU miss 分母。

策略命中率更高但 throughput/λ_pass 更低时，检查保存/回载、策略处理、长输入重算和排队，而不直接宣称缓存获益。ARC 保留热点也不意味着避免了所有冷 store。

Profiler 只对代表性高/低容量或切换样本另跑诊断，不开启于正式性能计时；累计 copy/operator 时间不能直接相加成请求延迟。三次运行报告每轮指标、中位数及范围，不声称统计显著；小差异进入额外配对复测。

## 8. 实现和交付

需新增统一 eviction_policy 接口（lru/lfu/arc），将现有 admission_policy 与替换策略解耦；同一缓存索引、读写路径、容量预留、访问去重和生命周期协议供三个策略使用。将启动脚本 hardcoded prefix 参数、客户端长度标记、场景生成器、capacity runner 的 reset/搜索/积压判断和报告审计按本方案扩展。

测试覆盖真实风险：ARC 状态不变量和 ghost hit、LFU 并列/淘汰重置、不同 request 去重、缓存重置、预算预留、完整快照、淘汰与加载的生命周期、回载输出一致性。GPU-only 与旧 LRU 的小型回归用于发现公共路径改动影响，历史结果不直接复用作新主矩阵对照。

每轮保存 config、environment、server log、trace manifest、request results、原始指标、policy events。计划输出到新的 `results/cache-policy/<batch-id>/`，汇总为新的批次报告并在 `results/EXPERIMENT-REPORT.md` 增加条目；预实验、正式吞吐、探索速率、正式边界、profile 分别计数，避免重复。

建议图表：各场景/长度下 capacity×policy 的闭环吞吐与 λ_pass；同一到达率下的 TTFT P95/Goodput 曲线；S2 扫描恢复、S3 热点切换的命中率/队列时间线；ARC p 与 LFU 频次残留；缓存读写量与策略维护成本。

## 9. 执行规模与顺序

1. 完成统一策略实现、确定性 trace 校验、生命周期检查。
2. 跑 14 配置预实验与长请求契约检查，冻结最终工作集/容量/阈值。
3. 跑 56 配置 × 3 次闭环正式测量。
4. 每配置探索和细扫，然后做最高通过点/相邻上点的长窗口确认。
5. 核心矩阵完成后，再跑 8 配置长输出扩展；混合输入、256 MiB 中间 CPU 容量和可变 prefix 长度另列后续批次。

完整 56 配置若每个确认两个速率点、各三次、每次 300 秒，仅发送窗口就约 **28 GPU 小时**；这还不包含吞吐实验、探索、启动、预热、冷却和排空。实际可能因低速最少请求数、区间不稳定或多次回退更长。预实验与正式结果分批交付，不用几十个请求的 smoke 冒充最高容量。

未来若研究可缓存 prefix 长短（例如 128/512/1024 token），应另设等长 prefix 的独立批次，公布对象字节与容量比例；若在同一 trace 混合不同大小 entry，则需要明确 byte-aware ARC 变体或 block 粒度及完整前缀可读规则，不能沿用本方案的经典等大 entry ARC 名称。
