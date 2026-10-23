# S5 多书检索验收

2026-10-23：阶段验证通过；在 S4 验收之后实施，S6 在本报告之后开始。

- `books: list[str]` 与原有 `book: str` 互斥；单字符串及单元素列表的真实结果一致。空列表返回空结果，不能误扩大为全库。
- 显式多书查询在语言/workspace/体裁过滤后的匹配书中预留每书一条；其余按全局相关度补齐，总数仍受 `top_k` 限制。`top_k` 小于匹配书数时返回明确参数错误。
- 候选仍使用原有全局融合与一次 rerank；预留发生在同一候选预算内，不会每本书各跑一次重排。构造一书强势占据前20名的测试，仍在20条候选、一次重排内覆盖第二本书。
- `group_by_book=true` 按 source 生成分组 Markdown；Web 保持原有 `results` 列表并附加 `groups`。每条仍包含原文、source、section、position，不生成对比结论。
- AGENTS 与 MCP playbook 同步加入跨书取材、各自定位章节、外层 Agent 综合的纪律。
- 使用《投资最重要的事》与《诺贝尔经济学奖经典收藏版》，查询“风险、不确定性与投资决策”：rerank 开/关均覆盖两书，单书兼容、真实 MCP 和 Web 调用通过。完整结果见 `s5-evidence.json`。
- 全量测试 **289 passed**，`ruff check src tests scripts` 通过。
