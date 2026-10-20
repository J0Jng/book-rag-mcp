# Implementation Brief: Book-RAG — 向量化前系统化 front/back matter 清洗 pipeline

## Background

当前 `parser → splitter → embedding` 链路上，非正文内容（front matter / back matter）的清理是**零散、格式内嵌、逐例补丁**的：

- `EpubParser` 内部有 4 处各自为政的清洗：`_EPUB_SKIP_NAMES`（按文件名跳过 nav/ncx/cover…）、`_is_toc_page`（链接密度启发式）、`_strip_contents_prefix`（文档开头 `Contents` 页眉）、`_is_dedication`（无目录映射 + 献词特征）。
- 这些逻辑**只有 EPUB 能触发**（依赖 spine / nav / 文档边界）。Markdown / TXT / PDF 解析出的 front/back matter **完全没有清洗**。
- 判定挂在"目录树映射"这类容器信号上（`mapped is None`），换个书源或文件格式就失效——这正是"逐个补丁"的根因。

实测现状（`books/` 真实书《斯坦福大学人生设计课》，`EpubParser` 现输出 59 个 section）：
- 已修复：献词页 `chapter1.xhtml`（P0.3 已剥离）。
- **仍混入索引**：`推荐序`(pos 0, 3546 字)、`致谢`(pos 58, 2159 字)。
- 边界正确保留：`引言`(13223 字)、`结语`(6739 字)、正文 11 章 + 节。

`docs/ROADMAP.md`「后续设想」第 4 条即本任务：**在 parser 输出之后、向量化之前挂一层系统化清洗**，与多书引入同步做。

## 目标

新增一个**格式无关、纯规则、确定性可测**的清洗层，统一在 `build_book_index` 的 `load_book()` 之后、语言检测 / `split_book()` 之前执行：

1. 对**任意 parser**（EPUB / PDF / TXT / MD）输出的 `ParsedBook` 生效，不依赖容器信号；
2. 判定 = **标题关键词（含副题）+ 位置带约束 + 保留词优先级**；
3. 剥离结果可审计：写 `cleaning_report.json` 到索引目录；
4. 绝不把整本书清空（fail-open）；
5. 零新增依赖、零 LLM 调用、幂等。

## 已确认决策（Joe 2026-10-20 拍板，不可违背）

| # | 决策 | 结论 |
|---|---|---|
| 1 | front matter 策略 | **保守**：只剥明确无检索价值项（封面/扉页/版权页/献词/推荐序/名家推荐/出版说明/目录残留）；**前言·序言·引言·导读一律保留** |
| 2 | back matter 策略 | **保守**：只剥（致谢/参考文献/参考书目/索引/译名对照表/作者简介/封底）；**结语·后记·附录一律保留** |
| 3 | 判定信号 | **纯规则**（标题关键词 + 位置约束 + 篇幅/边界规则）；**不引入 LLM** |

## 设计

### 1. 挂点（唯一改动入口）

`src/book_rag/rag/index.py` 的 `build_book_index()`：

```python
parsed = load_book(book_path)
parsed, report = clean_matter(parsed)          # ← 新增：向量化前清洗
language = parsed.language or _sample_language(parsed.sections)
```

必须放在**语言采样之前**（推荐序/致谢会污染语言检测样本）。

清洗后 `parsed.sections` 即唯一事实来源，后续 `split_book` / `build_toc` / `summarize_sections` / `build_concept_entries` 全部自动受益，无需改动这些模块。

### 2. 新模块 `src/book_rag/io/cleaning.py`

对外只暴露：`clean_matter`、`MatterPolicy`、`MatterReport`、`RemovedSection`。全部纯函数式：**输入 `ParsedBook`，返回新 `ParsedBook`（不原地修改入参）+ 报告**。

```python
@dataclass(frozen=True)
class MatterPolicy:
    strip_front: bool = True
    strip_back: bool = True
    position_ratio: float = 0.15    # front 只看前 N%、back 只看后 N%
    min_band_sections: int = 3      # 位置带下限：短书也至少覆盖前/后 3 个 section

@dataclass(frozen=True)
class RemovedSection:
    path: str
    kind: str          # "front" | "back"
    reason: str        # 命中的关键词，如 "title:推荐序"
    position: int      # 原始 position
    chars: int

@dataclass
class MatterReport:
    removed: list[RemovedSection]
    kept_sections: int
    total_sections: int
    aborted: bool      # True = 全部命中，放弃清洗（fail-open）
    policy: MatterPolicy
    def to_dict(self) -> dict: ...
```

### 3. 判定算法（实现按此顺序，顺序即优先级）

**Step 0 — 取标题**
`leaf = section.path.split(">")[-1].strip()`（path 是 `"章 > 节"` 链，只判定最末级标题，避免父路径关键词误伤）。
`path` 为空时跳过判定（保留）。

**Step 1 — 归一化 `_normalize_title(leaf)`**
- `unicodedata.normalize("NFKC", ...)`（全角→半角）
- 去 Markdown 装饰符 `#*_~`` 、首尾空白
- 折叠内部连续空白为单空格
- `casefold()`
- 去首尾 `：:：、,，.。·—–-|《》<>()（）[]【】` 等标点

**Step 2 — 保留词表优先（KEEP 命中即返回不剥）**
KEEP 命中 → 该 section **永不剥离**，无论位置。KEEP 词表必须包含（中英双语，精确匹配或"词 + 分隔符 + 副题"）：

```
前言 序言 自序 译者序 序 导读 导言 引言 编者按 楔子 引子 开篇
结语 尾声 后记 跋 致读者 结论
附录 术语表 词汇表 注释 尾注 脚注
preface foreword introduction prologue epilogue afterword conclusion
appendix appendices glossary notes
```

> 注意：`序` 单独保留（含"序言/自序/译者序"），`推荐序` 属于剥离侧——靠 Step 3 的整题/前缀判定区分，不会互相污染。

**Step 3 — 剥离词表匹配**
两张表，`FRONT_TITLES` / `BACK_TITLES`。匹配规则（对归一化后的 leaf）：

- **整题精确相等** → 命中；
- **前缀命中且关键词后紧跟边界符**（`空格 / : ： ｜ | - — – · 、 （ (` 或字符串结束）→ 命中。
  - 这样 `"推荐序 你的人生是工业品，还是艺术品？"` 能命中 `推荐序`；
  - 而 `"致谢的力量"`、`"版权经济学"` 这类词根相连的标题**不会**命中（无边界符），是防误伤的关键。

`FRONT_TITLES`（剥离）：
```
封面 封底 封面页 封底页 扉页 内封 书名页 版权 版权页 版权信息 出版说明 出版信息
献词 题献 献辞 目录 目錄 目次
推荐序 推荐语 名家推荐 推荐 赞誉 序一 序二 代序
cover title page half title copyright imprint dedication epigraph
praise endorsements table of contents contents toc
```
> `序一 / 序二 / 代序` 属于他人所写推荐序序列，剥离。`推荐` 单独时按整题匹配（`"推荐"`）才命中。
> 注意 `foreword / preface / introduction` **不在**本表——它们在 KEEP 中。

`BACK_TITLES`（剥离）：
```
致谢 鸣谢 谢辞 参考文献 参考书目 引用文献 文献 索引 人名索引 主题索引
译名对照表 作者简介 作者介绍 关于作者 封底 勒口 出版信息
acknowledgments acknowledgements references bibliography works cited
index name index subject index about the author about the authors colophon
```
> `文献` 单独命中风险：正文标题「文献综述」不含边界符不命中；`文献` 整题才剥。可接受。

**Step 4 — 位置带约束**

```
n = len(sections)
band = max(policy.min_band_sections, ceil(n * policy.position_ratio))
front_ok = section.position < band
back_ok  = section.position >= n - band
```

- front 命中 + `front_ok` → 剥离；否则**保留**（宁可留噪声，不可删内容）。
- back 命中 + `back_ok` → 剥离；否则保留。
- `policy.strip_front` / `strip_back` 为 False 时整类不剥（供 A/B 与排障，默认 True）。

**Step 5 — fail-open（不空书）**
若 `removed` 覆盖了全部 section（剥离后为空），**放弃本次清洗**：返回原 `parsed`（sections 原样）+ `report.aborted=True`，并在报告里保留候选记录。绝不抛错、绝不产出空索引。

**Step 6 — 重建 position**
剩余 sections 按原阅读顺序重建 `position = 0..k-1`（`dataclasses.replace`），返回新 `ParsedBook`（`title` / `language` 原样传递）。

### 4. 报告落盘

`src/book_rag/rag/index.py` 在 `save_structure(...)` 之后写：

```python
(idx_dir / "cleaning_report.json").write_text(
    json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
```

`to_dict()` 输出：

```json
{
  "policy": {"strip_front": true, "strip_back": true,
             "position_ratio": 0.15, "min_band_sections": 3},
  "total_sections": 59,
  "kept_sections": 57,
  "aborted": false,
  "removed": [
    {"path": "推荐序", "kind": "front", "reason": "title:推荐序", "position": 0, "chars": 3546}
  ]
}
```

### 5. 开关（逃生口，不改变默认行为）

`src/book_rag/config.py` 的 `BookRagConfig` 增加：

```python
clean_matter: bool = True
```

`from_env()` 读取 `BOOK_RAG_CLEAN_MATTER`：值为 `0 / false / off / no`（不区分大小写）时置 False，其余（含未设置）为 True。默认永远清洗——符合"不管什么情况下都要清洗"。

`index.py` 中：
```python
if config.clean_matter:
    parsed, report = clean_matter(parsed)
else:
    report = None
```
`report` 为 None 时不写 `cleaning_report.json`。

### 6. 与既有 parser 内清洗的关系（明确边界）

**保留** `_EPUB_SKIP_NAMES` / `_is_toc_page` / `_strip_contents_prefix` / `_is_dedication` **原样不动**：

- 它们处理的是 **容器级 / 文档级** 信号（spine 顺序、nav 映射、HTML 链接密度、页眉行），新 pipeline 看不到这些信息；
- 两者不冲突、不重叠：pipeline 只看 section 标题，让 `chapter1`（无名献词页回退名）自然落空，不会误判；
- 本次**不迁移**、**不删除**任何 parser 内逻辑（避免回归风险），pipeline 是"新增的第二道系统化防线"。

## Change List

### 1. 新增 `src/book_rag/io/cleaning.py`
上述 Step 1–6 全部实现，含 `_normalize_title`、`_match_title`、`FRONT_TITLES` / `BACK_TITLES` / `KEEP_TITLES` 常量、`clean_matter()`。

模块 docstring 要写明：格式无关、纯规则、挂点在 parser 之后 splitter 之前、fail-open 语义。

### 2. 修改 `src/book_rag/rag/index.py`
- `from book_rag.io.cleaning import clean_matter`
- `build_book_index()` 中按「设计 1」插入清洗调用；按「设计 4」写报告（`report is not None` 时）。
- 更新该函数 docstring 的流程描述（`load_book → clean_matter → 检测语言 → split_book → ...`）。

### 3. 修改 `src/book_rag/config.py`
- `BookRagConfig` 增加 `clean_matter: bool = True`
- `from_env()` 解析 `BOOK_RAG_CLEAN_MATTER`（见「设计 5」），并在 docstring 中补一行说明。

### 4. 新增 `tests/test_cleaning.py`

至少覆盖下列 case（用 `tmp_path` 或直接构造 `ParsedBook`，**不**依赖真实书/模型）：

| 测试 | 断言要点 |
|---|---|
| `test_normalize_title_variants` | 全角/大小写/多余空白/标点/Markdown `#` 归一化正确 |
| `test_strips_front_matter_titles` | `推荐序` / `封面` / `版权页` / `推荐序 副题` 被剥 |
| `test_strips_back_matter_titles` | `致谢` / `参考文献` / `索引` / `作者简介` 被剥 |
| `test_keeps_meaningful_matter` | `前言` / `序言` / `序` / `引言` / `导读` / `结语` / `后记` / `附录` / `术语表` **不**被剥（KEEP 优先） |
| `test_no_false_positive_on_word_prefix` | `致谢的力量` / `版权经济学` / `文献综述` / `索引式学习` 不剥 |
| `test_position_band_rejects_mid_body` | 20 个 section 的书里，`致谢` 出现在 pos 10 且 n=20（band=3）→ **保留** |
| `test_short_book_band_floor` | 3 个 section 的书，`推荐序`(pos 0) 与 `致谢`(pos 2) 都被剥（min_band_sections 兜底） |
| `test_positions_renumbered` | 剥离后 position 严格 `0..k-1` 连续递增 |
| `test_fail_open_when_all_removed` | 全书只有 2 个 section 且都命中 → `aborted=True`，sections 原样返回、非空 |
| `test_policy_switches_off` | `strip_front=False` 时 front 类不剥，back 类照常 |
| `test_idempotent` | 对 `clean_matter` 的输出再跑一次 → 无新增剥离、section 数不变 |
| `test_input_not_mutated` | 传入 `parsed.sections` 的 position 不被原地修改 |
| `test_report_fields` | `to_dict()` 含 policy/kept/total/aborted/removed，`removed[i].chars` 与原文长度一致 |

### 5. 修改 `tests/test_index.py`
新增 **1 个** 集成测试（复用现有 `config` fixture 与 `FakeHashEmbeddings`）：

- `test_build_book_index_writes_cleaning_report`：在临时 library 放一个含 `推荐序` + 正文 + `致谢` 的 `.md` 书，`build_book_index` 后断言：
  - `idx_dir / "cleaning_report.json"` 存在且 `aborted is False`；
  - `removed` 的 path 集合 == `{"推荐序", "致谢"}`；
  - `meta.chunk_count` 与清洗后正文一致（不含这两节文字）；
  - `toc.json` 顶层兄弟节点里**没有** `推荐序` / `致谢`。

## Acceptance Criteria（必须全部实际运行通过）

> Windows + MSYS：先 `export PYTHONPATH=` 清 Hermes 注入，用项目解释器 `.venv/Scripts/python.exe`。

1. **全量测试**
   ```bash
   cd "F:/code/projects/Book-RAG" && export PYTHONPATH= && .venv/Scripts/python.exe -m pytest tests/ -v
   ```
   期望：**基线 127 passed 不回归**，加上新增 test_cleaning.py（≥13 个） + test_index.py 新增 1 个，全绿、0 failed。

2. **真实书回归（用临时索引目录，不动 `books/.book_rag_index`）**
   写脚本 `scripts/`（或临时文件）执行：`BookRagConfig(library_path=books, index_dir=<临时目录>)` + `build_book_index(真实 EPUB, config)`，断言：
   - `cleaning_report.json` 的 `removed` 恰好包含 `推荐序`(pos 0) 与 `致谢`(pos 58)，`aborted=false`；
   - 清洗后 section 数 = **57**（59 − 2）；
   - `toc.json` 顶层节点为：`引言 / 第1章 … 第11章 / 结语`（无 `推荐序`、无 `致谢`）；
   - `引言`、`结语`、全部 11 章均在（KEEP 生效）。
   - 该步骤需真实 bge 模型（已缓存）；脚本内**先** `os.environ["HF_HUB_OFFLINE"]="1"` 再 import。
   - 完成后删除临时目录。

3. **幂等 / 空书保护自查**：`test_fail_open_when_all_removed` + `test_idempotent` 显式通过（在报告中引用测试名）。

4. **代码质量**：`.venv/Scripts/python.exe -m ruff check src tests` 无新增告警（项目已有 ruff 配置/缓存）。

## Explicitly Do NOT Do

- ❌ 不改 / 不删 / 不迁移 `parsers.py` 里任何既有清洗逻辑（`_EPUB_SKIP_NAMES`、`_is_toc_page`、`_strip_contents_prefix`、`_is_dedication`）。
- ❌ 不加 LLM 判定、不加 embedding 判定、不调网络（离线可用）。
- ❌ 不改 `splitter.py` / `retriever.py` / `structure.py` / `summarizer.py` / `concepts.py` / MCP / web / CLI 参数。
- ❌ 不改 `pyproject.toml` / `uv.lock`，不新增依赖。
- ❌ 不做"内容语义"级清洗（正文中间的广告页、页眉页脚）；本次只做 **section 级标题 + 位置**判定。
- ❌ 不动 `books/.book_rag_index`（真实索引重建由 Joe 决定），不改 `refresh_all_indexes` 的增量策略。
- ❌ 不硬编码任何具体书名 / 具体书的章节名到源码。
- ❌ 不在源码里写死`position_ratio` 以外的魔法数字（阈值一律具名常量或 policy 字段）。
- ❌ 不让 `clean_matter` 原地修改传入的 `ParsedBook`。

## 部署注意（写给 Joe，不是执行项）

`refresh_all_indexes` 以文件 hash 判定增量，**书没变就不会重建索引**——清洗层上线后，已有索引必须手动重建才会生效：

```bash
book-rag-mcp index          # 仅会重建新增/变更的书
# 需要全量重建时：删除对应 <index_dir>/book-* 子目录 + manifest.json 后重跑
```

本次不改增量策略（避免越过授权边界）。是否增加 `--force-reindex` 留作后续单独评估。

---

## 补丁记录：TXT / PDF 缺口（2026-10-20）

首版实现后，题目集（`tests/test_matter_scenarios.py`）用两道 strict xfail 暴露了缺口：
TXT 无章节标记、PDF 页级 path 时清洗层拿不到标题信号，推荐序 / 版权页 / 致谢仍进索引。
Joe 决定补齐，实现相对本 brief 有两处**有意扩展**（原「显式不做」清单中的第 5 条不再适用）：

1. **词表下沉**：`KEEP/FRONT/BACK_TITLES` + `normalize_title` + `match_title` 移到新模块
   `src/book_rag/io/matter_titles.py`（零依赖）。原因：`TextParser` 需要词表，
   而 `cleaning.py` 已 import `parsers`，词表留在 cleaning 会形成循环 import。
   `cleaning.py` 改为 re-export 同名符号，公开 API 不变。
2. **TXT**：`TextParser` 新增 `_is_matter_heading`——整段是单行、长度 ≤ 30、归一化后命中
   matter 词表时，视作章节边界切出独立 section，交由 `clean_matter` 判定。
   KEEP 词（前言 / 结语 / 附录）同样切出来，因此不会被并进相邻章节正文。
3. **PDF**：`clean_matter` 新增弱信号回退 `_content_heading`——当末级标题无信息
   （空 / 页码 `第 N 页` / 书名本身）时，取正文首行判定：
   整段 ≤ 2000 字、首行 ≤ 60 字（英文版权行更长）、首行整行或首个短语以关键词开头。
   弱信号同样遵守 KEEP 优先与位置带，`reason` 记为 `content:<关键词>` 以便审计。
4. **题集**：题 3 / 题 4 由 strict xfail 转为正式通过；新增题 13（弱信号受位置带约束）、
   题 14（弱信号同样 KEEP 优先）。

回归结果：`156 passed, 14 deselected`（快速层）+ `5 passed`（`-m integration`）；
真实 EPUB 仍 59 → 57；TXT / PDF 真实形态分别验证剥离生效。
