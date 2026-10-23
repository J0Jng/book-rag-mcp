# S3 收尾验收报告

> 本报告保留S3完工时的历史快照；后续S4–S6已实施，当前状态见 [S4–S6报告](s4-s6-acceptance.md)。

日期：2026-10-23。**S3 已通过验收并切换正式索引；S4–S6 未开始。**

## 实现与数据状态

- 五本 Markdown 均使用 `BAAI/bge-m3`，共 **3,257 chunk**：社会学 271、投资 322、人生设计 295、小说 521、套装 1,848。
- 正式目录：`books/.book_rag_index/`。原混合索引完整保留在 `books/.book_rag_index_before_s3_20261023/`，其中旧 EPUB、误建词典索引与残留目录没有混入正式目录；该备份仍是脏索引，不应直接恢复用于正式检索。
- 原始 EPUB 五份未移动或改写，执行前后的 SHA256 全部一致。Markdown 原文 hash 与新 manifest/meta 完全一致。
- 书籍发现排除库根 `jieba_dict.txt`。词典补充原文实际译名「阿迪克斯」，不更改正文；词典内容变化只重建 BM25。
- 检索保持每书向量/BM25 各 20 候选、全局 RRF(k=60)、最多 20 候选全局单次重排、top_k=5；没有减少候选数以满足延迟。
- 新增 `sections.json` 保存清洗后原文。第四章 23,426 字实际返回 15,000 / 8,426 字两页，拼接逐字一致，解决旧 chunk overlap 重复。
- 单书临时目录构建并读回验证后发布，逐书原子保存 manifest；支持中断续建、缺失文件修复、失败保留旧版本。Windows 文件/目录替换采用有界重试，永久失败仍报错。
- 缓存签名含 index.pkl、FAISS、meta、BM25 的 mtime 和词典指纹；同一模型的一次查询只编码一次。MCP 返回 source/section/position，MCP/Web 区分 rrf/rerank 排序分，不显示概率百分比，不套用统一 0.5 阈值。

## 真实模型与延迟

环境：Windows 11，Python 3.12.13，PyTorch 2.13.0+cpu，Intel64 Family 6 Model 197 Stepping 2，16 个逻辑 CPU，无 CUDA。

- 官方 reranker：`BAAI/bge-reranker-v2-m3`，revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`。
- 官方权重 SHA256：`d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286`；分段下载逐段检查 HTTP 范围与长度，合并后整文件校验，验收脚本再次核对实际缓存文件 hash。
- FP32 默认 16 线程实测约 8.1–8.3s，中位数未达标；4/8/12/16 线程试验仍未满足预算。该基线保留在 `s3-fp32-baseline.json` 与 `s3-thread-tuning.json`。
- 最终 CPU 默认使用同一权重的动态 INT8 Linear 层、最多 12 个 PyTorch 线程；CUDA 保持原生精度。`BOOK_RAG_RERANK_INT8=0` 可保留 FP32，`BOOK_RAG_RERANK_CPU_THREADS` 可指定正整数。线程设置影响进程共享的 PyTorch CPU 线程池。
- 以下为固定八题、五本书全部参与、每模式预热后五次的结果；冷索引加载时间与全部逐次耗时也保存在 JSON。模型加载单独计时，不混入预热查询。

| 查询 | 纯向量字面命中@3 | Hybrid 命中@3 | 重排命中@3 | 关闭重排：中位/最大秒 | 开启重排：中位/最大秒 |
|---|---:|---:|---:|---:|---:|
| 第二层次思维 | 3 | 3 | 3 | 0.049 / 0.051 | 3.142 / 3.452 |
| 安全边际 | 2 | 2 | 2 | 0.049 / 0.051 | 3.002 / 3.199 |
| 斯库特 | 3 | 3 | 3 | 0.051 / 0.055 | 3.610 / 4.081 |
| 阿迪克斯 | 3 | 3 | 3 | 0.050 / 0.051 | 3.667 / 3.899 |
| 有效市场假说 | 3 | 3 | 3 | 0.053 / 0.055 | 2.709 / 2.846 |
| 风险溢价 | 3 | 3 | 3 | 0.050 / 0.072 | 2.869 / 3.011 |
| 原型对话 | 2 | 3 | 3 | 0.052 / 0.079 | 3.181 / 3.640 |
| 奥德赛计划 | 3 | 2 | 3 | 0.053 / 0.078 | 2.927 / 3.267 |

预热后单次最大：关闭 **0.079s ≤1.5s**，开启 **4.081s ≤5s**。八题均 top-3 命中。每模式第一次查询各加载五份 FAISS；随后五次查询新增加载次数均为 **0**。每次重排只调用一次，候选数均不超过 20。

「原型对话」的 top-3 字面命中由纯向量 2 条增至 3 条；Hybrid 对「奥德赛计划」由 3 条降至 2 条，真实重排恢复为 3 条。不能据此宣称所有查询、所有书库均优于纯向量；这里只证明本次固定用例满足门槛。

引用样例（完整原文片段见证据 JSON）：

- 查询「原型对话」：source=`library/斯坦福大学人生设计课.md`；section=`斯坦福大学人生设计课 > 原型体验`；position=`167`。
- 查询「斯库特」：source=`library/杀死一只知更鸟.md`；section=`第一部 > 第七章`；position=`114`。

原计划示例「逆向选择」「模范少数族裔」「阿蒂克斯」在这批原文中未找到，明确记录为缺失，不伪造命中。使用原文真实词项与译名进行验收。

## 命令与真实验证结果

每次 Python/pytest 前先清空 `PYTHONPATH`。沙箱内曾遇到 Windows 临时目录权限及 asyncio socketpair 阻塞，以下完整测试、真实模型与接口验证在沙箱外执行，测试仅使用临时书库。

```powershell
$env:PYTHONPATH=''
& .venv/Scripts/python.exe -m pytest -q -o addopts= --basetemp=.pytest_cache/s3-all-final --tb=short
# 263 passed, 4 warnings in 22.74s（包括 14 项真实模型集成测试）
& .venv/Scripts/python.exe -m ruff check src tests scripts
# All checks passed!
& .venv/Scripts/python.exe scripts/verify_s3.py --index-dir books/.book_rag_index_s3 --output docs/reports/s3-evidence.json
# {"top3_all": true, "latency_all": true, "improvement_over_vector": true}
# 上条为切换前运行，目录现已移动到正式路径。
& .venv/Scripts/python.exe scripts/verify_s3.py --index-dir books/.book_rag_index --smoke-only --smoke-rerank --output docs/reports/s3-production-smoke.json
# 完整性及 MCP/Web 冒烟 passed；实际 score_kind=rerank，CPU dynamic-int8/12 threads。
```

正式切换前未发现 Book-RAG 服务进程，无需终止其他应用。切换后通过真实 MCP 客户端内存传输调用七个工具，并验证 Web 书籍、workspace、检索和后台刷新；五本书可见，fiction/finance/social/life 数量 1/2/1/1，刷新新建 0、跳过 5。未保留额外常驻服务，默认 CLI/MCP 启动入口会读取正式索引。

四条 warning 来自已有 langchain-community 及 PyTorch 量化接口弃用提示；当前版本实跑通过。本阶段未新增 torchao、未升级依赖。未来依赖迁移仍需单独评估，FP32 本机超时的限制未隐藏。

## 改动入口与证据

- IO/索引：`io/loader.py`、新增 `io/atomic.py`、`rag/index.py`、`rag/bm25.py`。
- 检索/重排：`rag/retriever.py`、`rag/reranker.py`；MCP/Web：`servers/mcp_server.py`、`web/index.html`。
- 验证：`scripts/verify_s3.py`，新增或扩充 atomic、index、hybrid、long_section、reranker、Web schema 测试；CI 静态检查包含 scripts。
- 文档：AGENTS/playbook、README、ROADMAP、DESIGN-TODO、扩充计划、历史执行队列和 IDEA 同步当前状态。
- 核心证据：`s3-evidence.json`（八题、逐次耗时、候选数、引用与缓存计数）；`s3-production-smoke.json`（正式路径完整性和重排接口）；`s3-originals-baseline.json`（原始文件 hash）。

## 未验证与后续边界

S4–S6 未开始：未调用在线摘要 LLM，未生成小说剧情梗概、跨书列表分组或主题聚类。GPU、多并发服务负载、更多书籍和长期稳定性不在本次验收范围；可选概念索引未构建时仍诚实提示缺失。没有执行 git commit/push。

发布后最终复跑：`pytest -q -o addopts= --basetemp=.pytest_cache/s3-release-final --tb=short` 返回 **263 passed, 4 warnings in 20.43s**，ruff 再次通过；验证脚本不传 `--index-dir` 的默认正式路径完整性检查也通过（`s3-production-integrity.json`）。已清理约 4.5 GB 的临时模型分段与合并副本，实际 HuggingFace 模型缓存、正式索引、旧备份均保留。
