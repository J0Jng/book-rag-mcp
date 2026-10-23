# 待设计清单（DESIGN TODO）

> 记录已决策、实施状态及后续讨论事项；已完成项不再作为未做功能清单。后续事项不是承诺或排期，实施与验收以 `docs/plans/` 和 `docs/reports/` 为准。
> 最后更新 2026-10-23。当前 S1–S6 已实现并完成阶段验证；主题混合/边缘成员等局限见 `docs/reports/s4-s6-acceptance.md`。

## 已决策不做（留档，避免重复讨论）

| 事项 | 决策 | 理由 |
|---|---|---|
| PDF 支持 | ❌ 不做 | PDF（尤其扫描版）需要 OCR + 版面还原，属于独立问题域；本服务只收能直接读取文字的格式（EPUB / TXT / Markdown） |
| 页码级引用 | ❌ 不做 | EPUB / TXT / MD 无稳定页码；能定位到章节 + position 已满足「找到具体章节和文段」的需求 |
| 生成式回答 | ❌ 不做 | 无状态检索服务，循环思考与答案生成交给外层 Agent |
| GraphRAG / 内置 LangGraph 循环 / query 改写引擎 | ❌ 不做 | 见 ROADMAP「明确不做」 |

## 设计与实施状态

### 1. Hybrid 检索（BM25 + 向量融合）

- **状态**：✅ S3 已实现并验收（2026-10-23），方案见 `docs/plans/book-rag-library-expansion-brief.md`，真实证据见 `docs/reports/s3-acceptance.md`。
  决策：BM25(jieba) + 向量 → RRF(k=60) → **融合后单次** cross-encoder 重排；向量模型收敛 bge-m3。
- **问题**：纯向量对「作者自造术语 / 精确命名 / 罕见专名」召回弱，改一个词就命中不了。
- **触发条件**：书籍库变多（多本同类书）、或用户反馈精确术语检索不到。
- **已落地**：jieba + ASCII 分词与可更新用户词典，RRF(k=60)，统一 bge-m3；进程缓存按索引文件 mtime 和词典指纹失效，不额外引入字符 n-gram 通道。
- **前置记录**：ROADMAP 已说明当初暂缓的理由（单书 311 chunk 时收益不抵成本）。

### 2. Rerank 精排（cross-encoder）

- **状态**：✅ S3 已验收。`BAAI/bge-reranker-v2-m3` 默认开启、全局融合后只重排一次；CPU 默认动态 INT8、最多 12 个 PyTorch 线程，可关闭量化。FP32 本机超出预算，保留了真实基线，不将其计为达标。
- **问题**：embedding 相似度 ≠ 相关度，top-K 里常有语义相近但答非所问的片段。
- **触发条件**：top-1 命中率在效果测试里出现明显失败样本。
- **验收门槛**：五本书、top_k=5，开启 ≤5s、关闭 ≤1.5s；章节内检索沿用同一流水线。超标不得以降级或减少书数冒充达标。

### 3. 多 workspace 分组

- **状态**：✅ S2 已实现并独立验收，方案见 `docs/plans/book-rag-library-expansion-brief.md` S2。
  决策：库根 `workspaces.yaml` 为元数据唯一真源（workspace + genre + tags，可多 workspace）；
  MD 只存纯正文；未声明的书归 `_unclassified` 并显式暴露。
- **问题**：现在一个 `BOOK_RAG_LIBRARY` 只能对应一个库；不同主题的书（技术 / 历史 / 文学）混在一起，检索会互相干扰。
- **触发条件**：书籍数量增长到跨领域共存、或需要按项目隔离。
- **已落地**：单库内元数据分组；MCP/Web 暴露 workspace 与 genre 过滤，Web 下拉切换；一本书可以属于多个 workspace，不复制正文文件。

### 4. 跨书综合 / 主题聚类

- **状态**：✅ S5/S6 已实现并实测（2026-10-23），方案见 `docs/plans/book-rag-library-expansion-brief.md`，证据见 `docs/reports/s5-acceptance.md`、`docs/reports/s6-acceptance.md`。
  决策：S5 只做「多书分组检索」（服务端不综合，架构 A1 不变）；S6 全量 chunk 离线 KMeans + LLM 命名主题，
  并**诚实标注**单层聚类无法完成真正的跨书主题对齐。
- **问题**：「对比 A 书和 B 书的观点」这类题现在只能由外层 Agent 多次检索后自行综合。
- **触发条件**：多书场景成为主要用法。
- **已落地**：显式多书范围与分组；复用全量向量的 KMeans，K 选择记录落盘；代表片段命名；只读 `search_by_topic`。运行时不调用 LLM、不综合回答。
- **后续待决策**：真正跨书主题对齐的二次聚类、边缘/短片段处理和主题纯度提升；本期没有把混合簇冒充语义对齐。

### 5. 摘要 / 概念层的本地 LLM 支撑

- **现状**：embedding 本地运行；S4 章节摘要/知识概念及 S6 主题命名可选调用 OpenAI 兼容在线 LLM。本次使用经授权的 DeepSeek 官方端点；在线 LLM 不进入检索请求路径。
- **问题**：若要求「整条链路零外网」，摘要层需要本地 LLM（Ollama / llama.cpp / vLLM）。
- **触发条件**：Joe 决定摘要层也必须本地化。
- **待决策**：本地模型选型与显存/内存预算、JSON 结构化输出的可靠性（对弱模型是主要风险）、每章摘要的耗时（几十章 × 每章一次调用）、是否需要「先只做概念抽取、不做摘要」的降级形态、与现有 `llm.py`（裸 urllib 调 OpenAI 兼容端点）如何复用——Ollama 有 OpenAI 兼容端点，理论上只需换 base_url + model。

### 6. 增量索引启动成本优化

- **现状**：每次启动（`refresh_all_indexes`）会对库内**每本书**算一次全量 sha256，未变则跳过。单本小书无感，但书库变大后启动时间线性增长。
- **触发条件**：书库规模到几十本以上，启动明显变慢。
- **待决策**：先比 `(size, mtime)` 再算 hash（快路径）、index 目录内记录 `hash` 供复用（省一次读全书）、或改为按需刷新（启动不扫，工具调用时才同步）。
