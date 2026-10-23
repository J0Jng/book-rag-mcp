# Implementation Brief: 书库扩充 — Markdown 归一 + workspace + Hybrid/Rerank + 跨书聚类

> 状态：**S1–S6 已实现并完成阶段验证**；S6 保留单层混合/边缘成员局限，详见验收报告。起草 2026-10-22　决策来源：Joe 三轮问答 + S0 结构体检实测
> 相关：`docs/reports/library-structure-inspection.md`（S0 报告）、`docs/DESIGN-TODO.md` #1/#3/#4、`docs/ROADMAP.md` 决策 5

> **2026-10-23 实施状态**：S1–S6 已完成阶段验证；正式索引五本书 3,257 chunk 未被 S4–S6 重算，新增 469 条摘要、2,908 条知识概念、8 个离线主题。S4 小说 31/31 章并核对三章，S5 实际双书查询与兼容验证通过，S6 全量向量复用、K 选择及只读主题工具实测通过；混合/边缘成员与跨书未对齐局限保留。见 `docs/reports/s4-s6-acceptance.md`。S3 延迟与旧混合索引备份说明见 `docs/reports/s3-acceptance.md`。

## Background

书库现在只有 1 本书（295 chunk、57 section、zh、bge-large-zh-v1.5），架构假设是「单库、单语言路由、纯向量检索」。
Joe 打算扩到 **5-10 本跨领域书籍**（小说 + 金融/经济/社会学知识类），这会同时踩中三件在 DESIGN-TODO 里挂着的事：
Hybrid 检索（#1）、多 workspace（#3）、跨书综合 / 主题聚类（#4）。

S0 结构体检（5 本实测，见报告）确认了三个事实，本 brief 的所有设计都基于它们：

1. **解析层没有灾难**：没有一本书退化成"整本一个 section"，最差 13 个 section → 不需要重写 parser。
2. **章节粒度偏粗**：`EpubParser` 一个 spine 文档 = 一个 section，而出版社常把一章拆成多个 HTML 小节。
   5/5 本单章都 >1 万字，最大 23,607 字（鲍曼《工作、消费主义和新穷人》第四章）。
3. **转 Markdown 是正确手段**：按 nav `#anchor` 切分实测能把最大章从 23,607 → 6,605 字（−72%），
   但 Joe 选择**只切到章级**（理由：目录树简洁、Agent 定位快）；因此长章问题改由**工具层保护**解决，而不是靠切分。

同时新需求是：**书籍最终统一以 Markdown 存储**，原始 EPUB/TXT 归档保留。
这让索引层从"三种 parser 并存"收敛为"一条 MarkdownParser 路径 + 一个离线转换器"。

## 已确认的设计决策（Joe 拍板，不可违背）

| # | 决策 | 影响 |
|---|---|---|
| 1 | **Markdown 为唯一存储格式**，`books/<workspace>/*.md` | 索引层只走 `MarkdownParser`；EPUB/TXT parser 降级为"转换器"内部件 |
| 2 | **`workspaces.yaml` 是元数据唯一真源**；MD 只存纯正文，不写 front matter | 一本书自描述能力为零，改元数据不必碰 MD 文件 |
| 3 | workspace 可**多对多**（一本书属多个 workspace），`genre` 分 `fiction` / `knowledge` | 检索要去重；摘要 prompt 按 genre 分叉 |
| 4 | **原始文件移到 `books/_originals/` 保留**（可回滚重转） | 数据移动操作，需 Joe 明确授权后执行 |
| 5 | **套装不拆**（《诺贝尔奖套装三册》按一本书处理），章节树里保留书名层级 | 跨书综合时它算 1 个 source，需在 tags 里标注"合集" |
| 6 | **章节切分只到章级**，不切小节 | 鲍曼书最大章 23,607 字保留 → 必须加 `get_section_content` 上限保护 |
| 7 | **向量模型收敛到 `BAAI/bge-m3` 单模型** | 现有《人生设计课》索引必须重建；跨书/聚类才有统一向量空间 |
| 8 | **Hybrid = BM25 + 向量 + RRF 融合** | 新增依赖 `rank-bm25` + `jieba` |
| 9 | **rerank 用 `BAAI/bge-reranker-v2-m3`，默认开**，单书快查可关 | 单次 ~3s（20 候选 CPU 粗估），内存 +2.12GiB |
| 10 | 小说要**章节级剧情梗概**（LLM 摘要层加 fiction 分支） | 每章一次 LLM 调用；依赖 `BOOK_RAG_LLM_*` 配置可用 |
| 11 | 跨书 = **全量 chunk 离线聚类 + LLM 命名主题**；本期先 **KMeans**，HDBSCAN 留后续 | 新增 `topics.json` + 主题检索工具 |
| 12 | 首批 **5-10 本，主要是 EPUB** | 不做大规模性能优化（分流/分片/分布式一律不做） |

**推翻的历史决策（需留档）**：`ROADMAP.md` 决策 5「Hybrid/rerank 暂缓」在单书 311 chunk 时成立；
现在书库跨领域扩张，其触发条件已满足，本 brief 正式取代该结论。

## 范围

**做**：Markdown 转换器、workspaces.yaml + 元数据、hybrid + rerank、索引缓存、genre 分支摘要、
多书检索与分组、离线主题聚类 + 主题检索工具、`get_section_content` 长章保护、
文档/playbook/ROADMAP/DESIGN-TODO 同步更新。

**不做（本期明确排除）**：
- 人物/地点/事件**实体索引**（Joe 选的是剧情梗概路线，不是实体抽取）
- 套装自动拆分；PDF/OCR；页码引用；服务端生成答案（架构 A1 不变）
- HDBSCAN / UMAP（KMeans 跑通验证后再评估）
- 多 `BOOK_RAG_LIBRARY`（workspace 是单库内分组，不是多库）
- 逐 chunk LLM Contextual Retrieval

## 分阶段任务

每阶段独立可交付、可验收；**前一阶段的验收不通过不得进入下一阶段**。

### S0 — 结构体检 ✅ 已完成

- 产物：`scripts/inspect_library.py`（只读）、`docs/reports/library-structure-inspection.md`
- 已产出：5 本诊断、红旗判定、anchor 切分收益实测。

### S1 — Markdown 归一（转换器 + 归档）

**状态：✅ 已完成（2026-10-22）**

**目标**：把 `books/_originals/` 里的 EPUB/TXT 转成 `books/<workspace>/<书名>.md`，章节结构固化在标题层级里。

**实际落地布局（与初稿不同，原因见下方偏差记录）**：

```
books/
├── library/           # 归一后的 Markdown（索引唯一输入）
│   ├── 工作、消费主义和新穷人.md
│   ├── 投资最重要的事.md
│   ├── 杀死一只知更鸟.md
│   ├── 斯坦福大学人生设计课.md
│   ├── 诺贝尔经济学奖经典收藏版.md
│   └── _conversion.json   # 源 → 产物映射 + 转换报告（幂等依据）
├── _originals/        # 原始 EPUB 归档（discover_books 跳过 `_` 前缀目录）
└── .book_rag_index/
```

> **偏差记录**：初稿写的是 `books/<workspace>/<书名>.md`，与「一本书可属多个 workspace」
> 冲突（多 workspace 会导致同一本书在多个目录各存一份）。改为**一本书一份 md 落在
> `books/library/`**，workspace 分组由 S2 的 `workspaces.yaml` 表达。

**实现要点**：
- 新增模块 `src/book_rag/convert/`（`markdown.py` + `__init__.py`），CLI 增加 `convert` 子命令
  （`book-rag-mcp convert [--source] [--dest] [--originals] [--copy-originals]`）；
- 标题层级：`#` = 书名/部（合集场景下是单本书名）/单层结构的章，`##` = 章，更深的目录节点降级为正文行；
- 合集（书名与章同为顶层节点）按「书名清单 + 标题模式」推断书名作用域，章挂到书名之下；
- 无 heading、不在目录中的文档（目录页 / 版权页 / 献词等）**一律不入正文**，原因 + 40 字预览
  写入报告，避免 5000 字目录污染章节；
- `discover_books` 新增跳过 `_` / `.` 前缀目录，防止 `_originals/` 里的 EPUB 被重复索引。

**实测结果（5 本书）**：

| 书 | 章数 | 正文字数 | 丢弃文档 | 最大章 | chunk |
|---|---|---|---|---|---|
| 工作、消费主义和新穷人 | 6 | 105,195 | 4 | 23,426 | 271 |
| 投资最重要的事 | 25 | 137,006 | 4 | 11,141 | 322 |
| 杀死一只知更鸟 | 31 | 220,903 | 3 | 11,911 | 521 |
| 斯坦福大学人生设计课 | 57 | 109,615 | 6 | 13,118 | 295 |
| 诺贝尔经济学奖经典收藏版 | 365 | 708,310 | 13 | 10,494 | 1,848 |

- 版权页 / 目录页文字在 5 份 md 中**零残留**（grep 侵权必究/版权所有/ISBN 均 0）；
- 章标题不再双写（含被拆成多行的 heading）；
- 合集产物：`# 自由选择（珍藏版）` → `## 珍藏版序言` → `## 第1章 市场的力量`，书名层级保留。

**验收（修订后口径）**：
1. ✅ `books/library/*.md` 全部生成，均可被 `MarkdownParser` 解析；
2. ⚠️ **无 🔴 这条不成立，且不打算靠切分解决**：《工作、消费主义和新穷人》第四章
   长 23,426 字（原 EPUB 就是单文档一章）。这是「只切到章级」决策的直接结果，
   属于已知取舍；**必须由 S3 的 `get_section_content` 上限保护兜底**，不得静默忽略。
   其余 4 本最大章 10.5k–13.1k 字，均在可接受范围。
3. ✅ 抽样核对：章标题与纸质目录一致（5 本逐一核对数量级）；
   正文总量与转换前差异 < 3%（如鲍曼 105,285 → 105,195，差 0.09%，为标题去重）。
4. ✅ 转换报告落盘：`books/library/_conversion.json`（源文件、产物、章数、字数、丢弃明细）。

**转 md 时发现并修掉的三个真实缺陷**（都会被真实书籍触发）：
1. `soup.get_text()` 会把 `<head><title>` 当正文吐出来 → 剥离 `head/title`；
2. 长度阈值 20 字把「第二章 归途」这类短章整章丢弃 → 降到 5 字；
3. 目录页 / 版权页无 heading 又不在目录中 → 原本会并进上一章，改为丢弃 + 记因。

### S2 — 元数据 + workspace

**状态：✅ 已独立验收（用户复核 201 passed，ruff 通过）。**

**目标**：`workspaces.yaml` 成为元数据真源，检索/列举/Web 全链路认 workspace 与 genre。

关键点：
- `workspaces.yaml` schema（草案，实施时可微调）：
  ```yaml
  workspaces:
    fiction:   {label: 小说, genre: fiction}
    finance:   {label: 金融投资, genre: knowledge}
    social:    {label: 社会学, genre: knowledge}
  books:
    "fiction/杀死一只知更鸟.md":
      workspaces: [fiction]
      tags: [美国文学, 经典]
  ```
- `BookIndexMeta` 扩展 `workspaces: list[str]` / `genre: str` / `tags: list[str]`（保持向后兼容读旧 meta.json）；
- **未声明的书 → `workspace: ["_unclassified"]`、`genre: "unknown"`**，必须在 `list_indexed_books` 里显式暴露，禁止静默归类；
- 改 `workspaces.yaml` 只需重写 meta，**不重跑 embedding**（元数据与应用解耦）；
- 多 workspace 的书：检索结果按 `(source, position)` 去重，workspaces 合并展示；
- 工具参数：`search_book_knowledge` / `list_indexed_books` 增加 `workspace` 与 `genre` 过滤；
- Web 界面加 workspace 切换（下拉），搜索与书籍列表都跟随。

**验收**：
1. `list_indexed_books` 按 workspace 分组输出，`_unclassified` 可见；
2. 同一 query 在 `workspace=fiction` 与 `workspace=finance` 下返回集合不交叉（除多 workspace 的书）；
3. 修改 yaml 后不重建索引即可生效（`pytest` 覆盖该路径）；
4. 现有一切测试 + 新增测试全绿；`ruff` 全绿。

### S3 — Hybrid 检索 + rerank + 索引缓存

**状态：✅ 已验收并完成正式索引切换（2026-10-23）；见 S3 验收报告。**

**目标**：精确术语/人名可召回；排序可信；延迟可控。

**架构约束（必须遵守）**：
```
query → 语言/workspace/genre 过滤选书
  → 每本：向量 top N ‖ BM25 top N
  → RRF 融合（k=60）成全局候选 top M
  → cross-encoder 重排 M 条（**只做一次**）→ top_k
```
- **禁止**"每本书各自 rerank 再合并"——延迟会随书数线性增长，违背 1-3s 预算；
- BM25 索引随书落盘（如 `bm25.json`/pkl），增量同步时一并重建；
- 分词：`jieba` + 可加载用户词典（小说人名/专名），英文走词切；具体克制方案见下"未决问题"；
- `BookIndexMeta.model_name` 统一为 `BAAI/bge-m3`；旧索引（bge-large-zh）需重建；
- **vector store 缓存**：进程内按 book key 缓存 `(faiss, meta, bm25)`，以文件 mtime/hash 失效；
- rerank 开关：`BOOK_RAG_RERANK=1/0`（默认 1），单书快查可关；
- 语言路由保留为**元数据**（用于展示与摘要 prompt），但不再决定 embedding 模型。

**验收**：
1. 精确术语用例（如"逆向选择""模范少数族裔"）top-3 命中原文 chunk，且优于纯向量（离线对比脚本记录）；
2. 小说人名检索（如"斯库特""阿蒂克斯"）top-3 命中；
3. 单次检索延迟（5 本书、top_k=5）：开 rerank ≤ 5s，关 rerank ≤ 1.5s（本机实测记录，允许放宽需说明）；
4. 第二次同 query 命中缓存，不重复 `FAISS.load_local`（可用 monkeypatch/日志断言）；
5. `pytest` + `ruff` 全绿。

**风险**：jieba 对虚构人名分词差 → 用户词典；m3 比 bge-large-zh 略慢；内存 3.33GiB + reranker 2.12GiB。

### S4 — genre-aware 摘要（小说剧情梗概）

**目标**：小说给出章节级剧情梗概；知识类保持现有框架/术语抽取。

关键点：
- `summarizer.py` 按 `genre` 分叉 prompt：`fiction` → 本章发生了什么 / 涉及人物 / 关键转折 / 悬念，
  `knowledge` → 现有 frameworks/terms/mental_models；
- 产出扩展：`summaries.json` 增加 `genre` 与剧情字段；概念索引只对 knowledge 类构建；
- 新增只读工具（如 `get_book_summaries`）返回章节梗概 + section 指针，**不生成新剧情**；
- LLM 不可用时整层降级跳过（沿用现有 fail-open 策略），**不得阻塞索引**。

**验收**：
1. 小说书 `summaries.json` 每章一条梗概，抽 3 章人工核对：梗概内容可在对应 section 原文中找到依据，无编造；
2. 知识类书的 frameworks/terms 抽取结果与改造前同量级（回归对比）；
3. 断网/无 key 时索引仍能建成，摘要层跳过且打日志；
4. `pytest` + `ruff` 全绿。

**前置**：`BOOK_RAG_LLM_*` 链路必须实测可用（当前《人生设计课》索引里**没有** summaries.json，说明历史上没通）。

### S5 — 跨书检索

**目标**：一次查询跨多本书，结果按书分组，服务端不做综合（架构 A1 不变）。

关键点：
- `book` 参数扩展为支持**列表**（或新增 `books`），并保持单字符串的向后兼容；
- 新增 `group_by_book: bool`：结果按 source 分组渲染（跨书对比题的原材料）；
- playbook 增加跨书纪律：先分书检索 → 再各自 `get_book_structure` → 最后自行综合。

**验收**：
1. `books=[A,B]` 返回结果同时覆盖两本书，且 `group_by_book=true` 时渲染成分组 Markdown；
2. 单字符串 `book="A"` 行为与改造前完全一致（回归测试）；
3. 跨书结果中每条都带 source/section/position；
4. `pytest` + `ruff` 全绿。

### S6 — 主题聚类 + LLM 命名

**目标**：离线产出 `topics.json`，支持"按主题取多书片段"。

关键点：
- 复用已构建的 chunk 向量（**不重新 embedding**）→ KMeans（余弦/归一化向量）；
- K 的选取：用 silhouette 或肘部法，落盘记录选择依据；
- 每个簇：LLM 命名 + 一句话描述 + 代表性 chunk 指针（严禁生成书中不存在的内容）；
- 落盘 `topics.json`（簇 → 名称/描述/成员 section 指针/count）+ 簇中心可单独建 FAISS；
- 新工具（如 `search_by_topic`）：按主题名/描述检索，返回按书分组的成员片段；
- **诚实标注局限**：单层 KMeans 只能得到"单书内/混合主题"，**真正的跨书主题对齐需要二次聚类（簇中心再聚）**，本期不做，需在工具输出与文档里写明。

**验收**：
1. 5 本书全量 chunk 聚类完成，K 选择有记录；`topics.json` 每簇有名称 + 描述 + ≥1 条指针；
2. 抽 3 个簇人工核对：成员确实同主题，名称不是编造；
3. `search_by_topic` 返回结果带 source/section/position 与所属簇；
4. `pytest` + `ruff` 全绿。

**风险**：KMeans 对 512 维向量簇语义可解释性差；LLM 命名质量不稳定；跨书对齐缺失（已声明的局限）。

## 需人工确认的风险操作（执行前必须 Joe 拍板）

1. **数据移动**：原始 EPUB/TXT → `books/_originals/`（S1）；
2. **索引重建**：删除 `books/.book_rag_index/*` 并用 bge-m3 重建（S3）——不可逆，需确认；
3. **新增依赖**：`jieba`、`rank-bm25`（S3）；`scikit-learn`（若未装，S6）；sentence-transformers 的 CrossEncoder（已在依赖内）；
4. **LLM 调用**：S4/S6 会真实消耗火山方舟额度（每章一次 + 每簇一次）；
5. **文档同步**：ROADMAP 决策 5 推翻、DESIGN-TODO #1/#3/#4 状态更新、AGENTS.md playbook 补跨书与 workspace 纪律。

## 未决问题（实施中需带回讨论）

1. **长章保护的具体形态**：`get_section_content` 超阈值（建议 15,000 字）时，是返回"前 N 字 + offset 续取"，
   还是返回"章节分段清单 + 提示用 search_within_section"？建议前者（可组合），待确认。
2. **BM25 中文分词方案**：`jieba` 默认 vs `jieba` + 字符 bigram 双路。小说人名是主战场，需小样本实测后定。
3. **rerank 是否参与 section 内检索**：section 内候选少（<50），rerank 收益可能不抵 3s 延迟，倾向不参与。
4. **套装书的 workspace 归属**：一个 source 覆盖三个领域，`tags` 里标注 `合集` 是否足够，还是允许 `genre: mixed`。
5. **`_originals/` 是否纳入 Syncthing 备份**：当前 `books/` 在 `.gitignore` 内，转换产物无版本历史。

## 整体验收（全部阶段完成后）

- 5 本书全部以 MD 入库，索引可复现（删索引重建结果一致）；
- `scripts/inspect_library.py books` 无 🔴；
- 精确术语、小说人名、跨书同题三类用例各有可复现的检索证据；
- `uv run pytest` 全绿、`uv run ruff check src tests scripts` 全绿；
- README / ROADMAP / DESIGN-TODO / AGENTS.md 与实现一致（不得留"支持 PDF""单库无 workspace"等过时表述）。
