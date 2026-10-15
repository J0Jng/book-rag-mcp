# Book-RAG 路线图与决策记录

> 记录方向性决策与「后续设想」，供任意会话（含飞书续接）对齐，避免重复讨论已拍板事项。最后更新 2026-10-15。

## 已拍板决策

1. **方向**：Book-RAG 定位为「book-to-skill 加强版 + 检索式 Agentic RAG」——减少语义损失，让外层 Agent 先定位章节、再精读原文、跨章节综合。
2. **架构 A1**：Book-RAG 保持无状态检索服务（不内置 LangGraph / agent 循环），循环思考交给外层 Agent；Book-RAG 只提供细粒度原子工具 + playbook 纪律。此架构已获 2026 年前沿论文（RIT-RAG / DeepRead）验证为「结构工具 + 外层 agent 循环」的 SOTA。
3. **不塞全文**：明确不采用「整本书塞进 LLM 上下文」方案。理由：单轮问答可行，但进入多轮讨论后全文常驻会迅速撑爆上下文窗口；多轮场景下「按需检索 + 只注入相关片段」不可替代，这是检索式 RAG 相对长上下文的硬差异。
4. **绝不编造**：所有返回内容锚定原文，检索不到明说「未检索到」，禁止用通识补全。
5. **通用检索增强暂缓**：Hybrid（BM25+向量）与 rerank 当前**不加**——单本书候选集小（311 chunk）、中文分词 + 跨书打分不可比的成本高、CPU rerank 延迟（约 2s/次）伤「无状态快检索」体验。

## 后续设想（触发条件：多本书 / 书种类增多）

当书籍库扩展到多本书、或书种类（技术 / 历史 / 多语言）增多时，重新评估并引入：

1. **Hybrid 检索**：`rank-bm25` + `jieba` 分词 + RRF 融合。解决「作者原命名 / 精确术语」在语义向量下失配的问题（book-to-skill 质量规则「保留作者精确性」的技术兑现）。
2. **Rerank 精排**：`bge-reranker`（cross-encoder）对 top-K 候选重排，解决「embedding 相似度 ≠ 相关度」。
3. **PDF 章节层级提取**：`PdfParser` 当前只有页级 path、level 恒 1（与 EpubParser 同类缺陷）；后续若大量引入 PDF 书，补书签 / 标题层级提取。

## 差异化定位

开放、可被任意 Agent（Claude Code / Codex / Cursor / Hermes）调用的**无状态 MCP 服务**，与闭源消费级产品（NotebookLM / PDF.ai / ChatPDF）错位：

- 本地零成本 embedding（bge），不依赖云长上下文；
- 按书增量索引、结构层地图（TOC）、可选降级的离线 LLM 摘要层；
- 多轮讨论中只注入相关片段，不撑爆上下文。

## 演进路线

`P0`（修 parser，进行中）→ `P1`（薄 LLM 摘要层 + 概念索引，可选降级）→ `P2`（playbook 文档 / AGENTS.md）

**明确不做**（除非触发上文条件）：GraphRAG、内置 LangGraph 循环、逐 chunk LLM Contextual Retrieval、query 改写引擎、跨书综合引擎。