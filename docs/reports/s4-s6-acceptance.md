# S4–S6 完成与验证记录

日期：2026-10-23。按S4→S5→S6顺序实施，每阶段先完成验证再进入下一阶段。已完成本期范围；没有执行git commit/push。

| 阶段 | 当前产物与实际证据 |
|---|---|
| S4 体裁摘要 | 小说31/31章；五书469条摘要；四本知识书2,908条概念；核对三章原文；知识回归框架6→7、术语16→13；[阶段报告](s4-acceptance.md) |
| S5 多书检索 | books列表、group_by_book、一次全局重排与每书覆盖；真实双书查询、单字符串兼容、MCP Markdown/Web JSON通过；[阶段报告](s5-acceptance.md) |
| S6 主题聚类 | 五书3,257个既有向量、8个主题；K选择有记录；所有成员指针核对；只读主题检索通过；[阶段报告与语义局限](s6-acceptance.md) |

新增只读MCP工具 `get_book_summaries`、`search_by_topic`，现共9个原子工具。运行时不调用在线LLM生成回答；离线命令 `summarize`、`topics` 支持恢复，拒绝不完整/过期产物。

**最终验证**：全量 `pytest -q -o addopts=` → **296 passed**（含14个真实缓存模型集成测试）；`ruff check src tests scripts`、`git diff --check` 通过。保留既有LangChain/PyTorch弃用警告。

真实MCP/ASGI冒烟：9个工具、5本书、workspace数量、原文检索、章节续取、概念与摘要读取通过；刷新新建/重建0、跳过5。见 `s4-s6-interface-smoke.json`。多书接口另见 `s5-evidence.json`，主题接口见 `s6-evidence.json`。

`s4-integrity.json`、`s6-evidence.json` 保存全量指针与核心索引不变的结果。原始5个EPUB的SHA256与S3基线一致，见 `s4-s6-originals-integrity.json`。S4–S6没有重算正文向量。

API key仅在调用进程内使用，未写入仓库、报告或配置。补建新书仍需通过进程环境配置 `BOOK_RAG_LLM_*`；现有产物与检索在无key时可用。Ark配额不足后，已按用户授权切换DeepSeek官方接口。

**实际局限**：主题名/描述只概括代表片段；单层KMeans仍有混合与边缘成员，三簇核对中已记录署名短片段，不能据此推导所有成员统一观点。真正跨书对齐未实现，按原计划留待二次聚类。摘要与标签是离线生成内容，回答前仍须回到原文核对。

可重跑（读取现有结果无需在线key）：

```powershell
.venv/Scripts/python.exe -m book_rag.cli summarize
.venv/Scripts/python.exe -m book_rag.cli topics
.venv/Scripts/python.exe scripts/verify_s4.py
.venv/Scripts/python.exe scripts/verify_s5.py
.venv/Scripts/python.exe scripts/verify_s6.py
.venv/Scripts/python.exe -m pytest -q -o addopts=
.venv/Scripts/python.exe -m ruff check src tests scripts
```

语义核对由Codex阅读原文完成，报告没有声明已经获得用户人工验收。
