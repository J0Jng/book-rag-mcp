# S4 章节梗概与知识摘要验收

2026-10-23：阶段验证通过。小说 31/31 章，四本知识类书摘要与概念索引完整；S5 在本报告之后开始。

| source | 摘要分组数 | 概念数 |
|---|---:|---:|
| 工作、消费主义和新穷人 | 7 | 96 |
| 投资最重要的事 | 25 | 294 |
| 诺贝尔经济学奖经典收藏版 | 349 | 2,139 |
| 斯坦福大学人生设计课 | 57 | 379 |
| 杀死一只知更鸟 | 31 | 不建立 |

总计 469 条摘要、2,908 条概念。套装书的 365 个解析 section 中有重复路径，按完整章节路径聚合后是 349 个摘要分组，所有 section 文本仍参与输入，未拆成多本 source。

- 使用经用户授权的 DeepSeek 官方 API / `deepseek-flash`，密钥仅存在调用进程环境，未写入项目、报告或凭据文件。Ark 实测月度配额耗尽的证据另存于 `s4-llm-diagnostic.json`，不再重复尝试该端点。
- 小说提取剧情、人物、转折、悬念与证据。人物名允许忽略排版空白后对齐，落盘恢复原文；段落编号由代码映射回真实原文，不能把改写引文当作原文。失败章节不进入完整发布结果。
- 成功章节写入 `summaries.pending.json`，输入哈希匹配时断点恢复；仅完整结果发布 `summaries.json`。状态包含书籍哈希和体裁，读取工具拒绝过期/不完整结果。显式 `summarize` 命令补建摘要和知识概念，不重新计算正文向量。
- 小说第 1、15、31 章由 Codex 阅读原文逐条语义核对，通过；这不冒充用户人工验收。详情见 `s4-semantic-review.json` 和 `s4-fiction-audit-inputs.json`。
- 三本知识书取相同实际章节，与 HEAD 中改造前的原版摘要实现对照：框架 6→7、术语 16→13，均在 0.5–2 倍范围内。详情见 `s4-knowledge-regression.json`。
- 无 key、网络失败、配额不足时摘要层记录日志并降级，核心向量/目录仍可构建；只读 MCP 不调用 LLM。
- 五本书的 manifest、核心 FAISS、docstore、BM25、目录和章节快照哈希未变；3,257 个正文 chunk 未重新 embedding。见 `s4-build.json`、`s4-integrity.json`。
- 真实 FastMCP 客户端覆盖 8 个工具，Web ASGI 管理/检索冒烟通过，刷新重建 0、跳过 5。见 `s4-interface-smoke.json`。
- 全量 `pytest -q -o addopts=`：**281 passed**；`ruff check src tests scripts`：通过。保留既有 LangChain/PyTorch 弃用警告。

摘要仍是用于定位章节的离线生成内容，回答与引用须回到原文核对；服务端不内置回答生成或 Agent 循环。
