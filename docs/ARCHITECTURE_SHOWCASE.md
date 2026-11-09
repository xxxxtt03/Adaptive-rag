# 🏛️ Adaptive-RAG 工业级系统架构白皮书 (System Architecture Showcase)

> **项目定位**：针对传统 FAQ 系统长尾命中率低、常规 Naive RAG 多跳推理能力弱与严重幻觉、以及大模型在线推理成本高与缺乏安全护栏等生产痛点，设计并实现的基于 **LangGraph 状态机 × 自适应多智能体（Adaptive-RAG）** 的工业级电商智能客服中台。

---

## 🗺️ 一、端到端系统核心请求拓扑流转 (End-to-End Flowchart)

```mermaid
flowchart TD
    Client["客户端 (Web UI / 移动端 / 微服务 RPC)"]
    Ingress["Nginx Ingress (proxy-buffering: off, 解决 SSE 流式卡顿)"]
    FastAPI["FastAPI 接入层 (Token 认证 / 速率限制 / 链路 TraceId 绑定)"]
    Router{"Prototypical Router\n(双轨意图与复杂度仲裁)"}

    Client -->|HTTP / SSE| Ingress --> FastAPI --> Router

    %% Fast-Path Short Circuit
    Router -->|CHITCHAT / GREETING| FastPath["毫秒级内存短路 (<10ms)\n(静态规则模板直出)"]
    FastPath --> Synthesize

    %% Semantic Cache
    Router -->|POLICY_FAQ| SemCache{"语义向量缓存\n(余弦相似度 >= 0.92)"}
    SemCache -->|🎯 缓存命中 (<15ms)| Synthesize

    %% Hybrid Retrieval Pipeline
    SemCache -->|Cache Miss| Hybrid["混合检索阶段 (Hybrid Retrieval)"]
    Hybrid --> BM25["BM25 词法召回 (Top-20)"]
    Hybrid --> Dense["Dense 稠密向量召回 (Top-20)\n(BGE-small-zh-v1.5)"]
    BM25 --> RRF["RRF 倒数排名融合 (k=60)"]
    Dense --> RRF

    %% Cascade Rerank & Graph-RAG
    RRF --> Prune["级联剪枝 (严格截断 Top-8 候选)"]
    Prune --> CE["Cross-Encoder 深度重排 (bge-reranker-base)\n[支持 INT8 动态量化, 耗时减半至 73ms]"]
    Prune --> GraphRAG["Graph-RAG 政策依赖图 (0.011ms)\n(DAG 拓扑仲裁, 置顶法律硬约束)"]

    %% ReAct State Machine
    Router -->|TOOL_REQUIRED / COMPLEX| Agent["LangGraph 多智能体协同状态机"]
    CE --> Agent
    GraphRAG --> Agent

    subgraph AgentStateMachine["LangGraph 状态机运行时 (Runtime)"]
        Planner["ReAct 规划器 (Prompt-template + DeepSeek-V3)"]
        Sanitize["Plan 净化器 (去重 / 白名单 / 步数硬熔断 max_steps=6)"]
        Executor["工具执行器 (22个原生工具 + 2个MCP工具)"]
        Fallback["READ_FALLBACKS 级联容错\n(track_shipment -> get_order -> search_knowledge)"]
        
        Planner --> Sanitize --> Executor
        Executor -.->|下游504/超时| Fallback -.-> Planner
    end

    %% HITL Guard
    Executor --> RiskCheck{"工具风险等级检查\n(ToolRisk)"}
    RiskCheck -->|READ / WRITE| ExecuteTool["执行业务工具逻辑"]
    RiskCheck -->|DESTRUCTIVE\n(取消订单/退款)| HITL["人机协同 (HITL) 拦截挂起\n• 生成 15min TTL Action Proposal\n• 下发 confirm_action_id 弹窗"]

    HITL -->|用户点击确认| CAS{"CAS 原子消费状态\n(防并发击穿与重放攻击)"}
    CAS -->|验证成功| ExecuteTool
    CAS -->|已消费/过期| InvalidReq["抛出 InvalidRequestError 拦截"]

    ExecuteTool --> Faithfulness{"忠实度与幻觉校验网关\n(Faithfulness Checker)"}
    Faithfulness -->|通过| Synthesize["最终答案合成与流式输出 (Synthesize)"]
    Faithfulness -->|幻觉漂移| Refuse["自纠错补检 / 优雅拒答转人工"]

    Synthesize -->|SSE 打字机逐字推送| Client
```

---

## 🛠️ 二、五大生产级二次开发业务引擎 (Secondary Development Blueprint)

我们在项目中深度扩展了 5 大电商高频真实业务工具，涵盖了不同安全等级与风控边界：

```
                            Adaptive-RAG 二次开发业务矩阵
                            
┌─────────────────────────┬─────────────────────────┬─────────────────────────┐
│ 1. 跨境关税计算器       │ 2. 增值税专票税号引擎   │ 3. 会员积分与卡券核销   │
│ (CalculateDutyTool)     │ (IssueInvoiceTool)      │ (LoyaltyPointsEngine)   │
├─────────────────────────┼─────────────────────────┼─────────────────────────┤
│ • 风险: ToolRisk.READ   │ • 风险: ToolRisk.WRITE  │ • 风险: READ & WRITE    │
│ • 9.1% 跨境综合税       │ • 18位统一信用代码正则  │ • 金卡9折/钻石8折兑换   │
│ • 23.05% 高档化妆品消费税│ • 订单已送达时效强校验  │ • 积分流水号与卡券码生成│
│ • 单笔5000/年26000限额  │ • 增值税合规防篡改声明  │ • 余额不足熔断拦截     │
└─────────────────────────┴─────────────────────────┴─────────────────────────┘
┌───────────────────────────────────┬─────────────────────────────────────────┐
│ 4. 地址解析与隐私脱敏引擎         │ 5. 最优叠券与互斥决策引擎               │
│ (ParseAddressTool)                │ (CalculateCouponStackingTool)           │
├───────────────────────────────────┼─────────────────────────────────────────┤
│ • 风险: ToolRisk.READ             │ • 风险: ToolRisk.READ                   │
│ • 《个保法》手机号脱敏: 138****5678 │ • 秒杀品类剔除满减券硬约束              │
│ • 提取后4位核验码 (phone_tail)    │ • 定制类商品剔除新客券硬约束            │
│ • 新疆/西藏等偏远改派加收8元运费  │ • 同类满减券贪心优选 + 运费券正交叠加   │
└───────────────────────────────────┴─────────────────────────────────────────┘
```

---

## ⏱️ 三、毫秒级算力开销剖析与量化优化 (Latency & Quantization Profiling)

在本地 CPU 基准测试环境下的实测耗时分布：

| 流水线组件 / 阶段 | 未量化基准耗时 (FP32) | INT8 动态量化耗时 | 耗时占比 | 优化机制与工程策略 |
| :--- | :---: | :---: | :---: | :--- |
| **1. 意图分类短路** | `< 10 ms` | `< 10 ms` | 4.7% | 内存正则关键词预匹配，打招呼与闲聊极速直出 |
| **2. 语义向量缓存** | `< 15 ms` | `< 15 ms` | 7.1% | 语义空间余弦相似度 $\ge 0.92$ 直接命中，避开后续重度计算 |
| **3. BM25 词法检索** | `0.795 ms` | `0.795 ms` | 0.4% | 基于内存倒排索引，精准抓取 SKU 实体与专有名词 |
| **4. RRF 秩次融合** | `0.336 ms` | `0.336 ms` | 0.2% | $k=60$ 排名倒数加权，平滑物理量纲差异 |
| **5. Dense 向量比对** | `26.68 ms` | `26.68 ms` | 12.6% | BGE-small 嵌入前向 + 余弦相似度比对 |
| **6. Graph-RAG 图遍历** | `0.011 ms` | `0.011 ms` | 0.01% | 内存 DAG 拓扑覆盖仲裁，置顶法律约束 |
| **7. Cross-Encoder 精排** | **`149.79 ms`** | **`73.25 ms`** | **71.1%** | **PyTorch 动态 INT8 量化，耗时减半 (2.04x 加速)，打分保真度 0.9781** |
| **全检索流水线总耗时** | **`210.81 ms`** | **`116.07 ms`** | **100%** | **稳稳收敛在 150ms 以内，满足大厂在线生产 SLA** |

---

## 🛡️ 四、高可用容灾与安全防御纵深 (Defense in Depth)

1. **死循环与状态机熔断**：
   * `max_tool_calls = 5` 与 `max_agent_steps = 6` 硬计数器拦截；
   * `_sanitize_plan` 执行前白名单过滤与连续重复调用去重；
   * 工具报错 Scratchpad 反馈机制，相同入参连续失败 2 次强制熔断并跳转至 `synthesize` 降级兜底。
2. **破坏性动作 HITL 安全令牌机制**：
   * `ToolRisk.DESTRUCTIVE` 动作强制挂起，生成携带 15 分钟 TTL 的 `action_id`；
   * 采用单次有效 Nonce 令牌（Single-Use），底层执行数据库条件更新：
     `UPDATE proposals SET status='CONSUMED' WHERE id=:id AND status='PENDING' AND expires_at > now`
   * 影响行数为 0 立即拒绝，杜绝网络连击导致的订单重复取消与越权击穿。
3. **输入边界提示词注入防御（Prompt Injection Guardrail）**：
   * 正则与语义层阻断：DAN 越狱模式、角色扮演脱敏逃逸、`<|im_start|>` 分隔符注入、系统设定套取与数据外泄攻击。
4. **长文本记忆预算管理（128k Token Budgeting）**：
   * 工具端只保留业务核心字段（Field Projection），知识库 Chunk 硬截断在 400 字符内；
   * 滑动窗口仅保留最近 3 轮完整问答，早前交互滚动压缩为 100 字摘要，将 Input Tokens 稳定压制在 3,500 Tokens 以内。

---

## 🚢 五、云原生高可用与全景可观测性 (Cloud-Native & Observability)

```
                            云原生监控与部署编排体系
                            
   ┌────────────────────────────────────────────────────────┐
   │                   Kubernetes 集群环境                   │
   │                                                        │
   │   [Nginx Ingress Controller]                           │
   │   • proxy-buffering: "off"  (解决 SSE 打字机流式卡顿)   │
   │   • proxy-read-timeout: "300" (防止 Agent 长推理超时)  │
   │               │                                        │
   │               ▼                                        │
   │   [Service: ClusterIP]                                 │
   │               │                                        │
   │       ┌───────┴───────┐                                │
   │       ▼               ▼                                │
   │   [Pod Replicas (HPA 自动弹性伸缩 2~10 副本)]          │
   │   • RollingUpdate (maxSurge: 1, maxUnavailable: 0)     │
   │   • 非 root appuser (UID 1000) 安全上下文              │
   │   • livenessProbe (/api/v1/health)                     │
   │   • readinessProbe (/api/v1/ready)                     │
   │   • preStop hook: sleep 10 (配合优雅停机 60s)          │
   └───────────────┬────────────────────────────────────────┘
                   │
                   ▼ /metrics 端点抓取 (15s 周期)
   ┌────────────────────────────────────────────────────────┐
   │             Prometheus + Grafana 监控大盘              │
   │                                                        │
   │   1. QPS 吞吐量 (按策略与状态分流)                     │
   │   2. Adaptive 路由决策分布饼图 (Direct/Agent/Multi)    │
   │   3. P50 / P95 / P99 端到端分位数延迟曲线              │
   │   4. 各阶段 P95 耗时分解 (Routing, Retrieval, CE, LLM) │
   │   5. 22 个业务工具调用频次与失败率统计                 │
   │   6. Token 消耗速率 (Input vs Output Tokens/sec)       │
   │   7. 忠实度检查幻觉拦截数监控                          │
   └────────────────────────────────────────────────────────┘
```

---

## 🎯 六、可复现指令与工程门禁速查表

| 验证目标 | 终端复制命令 | 预期成功标准 |
| :--- | :--- | :--- |
| **全栈 6 阶段体检门禁** | `python scripts/pre_flight_check.py` | 6 大门禁 100% 全部 [PASS] |
| **全量自动化测试套件** | `pytest tests/ -q` | **222 passed in 3.38s** |
| **7 轮真实 LLM 真机回归** | `python scripts/e2e_walkthrough.py` | 覆盖关税、发票、积分、HITL 拦截与防重放 |
| **INT8 量化加速评测** | `python scripts/benchmark_quantization.py` | **2.04x 加速**，打分保真度 0.9781 |
| **毫秒级检索耗时剖析** | `python scripts/profile_latencies.py` | 输出各阶段微秒级分布，确认级联剪枝必要性 |
| **高并发压力测试** | `python scripts/stress_test.py` | 多 Worker 并发统计 P50/P90/P99 与场景矩阵 |
| **离线评测消融** | `python -m eval.runner --task retrieval` | **MRR: 0.9321**, Recall@3: 96.2% |
| **一键双击极速启动** | 双击 `start.bat` 或运行 `start.bat` | 自动启动全栈并在就绪后弹出浏览器访问 |
