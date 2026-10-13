# Book-RAG 效果检验测试任务书

> **背景**：现有 52 个测试全部用 `FakeHashEmbeddings`（sha256 随机向量）+ mock `detect_language`
> 验证「管道不炸、格式对」，但**从未验证实现的效果**——语义检索是否真的返回语义相关原文、
> 真实语言路由是否有效。本任务书新增一层「效果检验测试」，用真实 embedding 模型与真实
> `langdetect` 验证实现是否真的"work"，而非仅仅"不崩"。
>
> **架构师实测结论（交付本任务书前已探针验证）**：
> - ✅ 真实 `bge-large-zh-v1.5` 语义检索 4/4 命中目标主题，相似度区分度清晰（0.57~0.69）。
> - ⚠️ `bge-m3` 加载约 134s（慢但可用），跨语言测试须 opt-in 且设 `HF_HUB_OFFLINE=1`。
> - 🐛 **已确认缺陷**：langdetect 把短中文判成韩语，导致默认语言过滤返回空。见文末「附」。

## 设计决策

1. **两层结构**：快速层（默认 `pytest` 运行，无模型下载）+ 集成层（`-m integration`，真实模型，opt-in）。
2. **测试隔离**：新增测试文件**不修改**现有 `tests/test_*.py`；`FakeHashEmbeddings` 帮助类在各自新文件内局部定义（沿用现有 test_index/test_retriever 的重复模式，避免触碰旧文件）。
3. **真实模型 gate**：集成测试运行前用 `huggingface_hub.try_to_load_from_cache(repo_id, "config.json")` 判断模型是否已缓存；未缓存则 `pytest.skip`（**绝不触发下载**）。同时 `autouse` fixture 设 `HF_HUB_OFFLINE=1`、`HF_HUB_DISABLE_SYMLINKS_WARNING=1`，保证离线确定、不联网。
4. **语义测试绕开语言过滤 bug**：语义质量测试调用 `search_books(..., search_all_languages=True)` 以隔离「embedding 质量」与「语言过滤缺陷」（后者由快速层回归测试单独覆盖），明确不在此处修 bug。
5. **缺陷记录**：短中文误判 bug 已由先决 ticket `book-rag-fix-short-query-langdetect.md` 修复；本任务书的快速层将该用例作为**普通断言**纳入（回归防护），不再用 xfail。

## 改动清单

### 1. `pyproject.toml`
在 `[tool.pytest.ini_options]` 内新增（不要动其它键）：
```toml
    addopts = "-m 'not integration'"
markers = [
    "integration: 效果检验集成测试（需已缓存真实 embedding 模型，默认跳过）",
]
```
> 注：`addopts` 默认排除 integration；运行集成测试用 `pytest -m integration`（CLI 的 `-m` 覆盖 addopts）。

### 2. `tests/test_effect_language_routing.py`（新建，快速层，无模型）
用 `FakeHashEmbeddings`（sha256 假向量，局部定义）但你**不 mock `detect_language`**，
验证「真实语言检测 → 真实语言过滤路由」链路的行为。用例：
- `test_long_chinese_query_routes_to_zh_book`：建一本中文书，长中文 query（如「人生设计的方法有哪些」）→ 结果非空且 `source` 均为中文书。
- `test_short_chinese_query_not_misrouted`：短中文 query（如「招聘」）→ 断言结果非空（依赖先决 ticket「短中文误判修复」落地，`detect_language` 已加 CJK 脚本启发式，此处为**普通断言**非 xfail）。
- `test_empty_and_numeric_query_returns_unknown_gracefully`：`detect_language("")`、`detect_language("123456")` 不抛异常，返回可预期结果。
- `test_mixed_language_query_no_crash`：中英混排 query 调用 `search_books` 不抛异常（结果可为空）。

### 3. `tests/test_effect_semantic.py`（新建，集成层，真实 `bge-large-zh-v1.5`）
`@pytest.mark.integration` + `_require_model` gate。在 `tmp_path` 写一本多主题中文书
（文件名 `生活百科.md`，内容见下方语料），用**真实** `refresh_all_indexes`（不 mock）建索引：
```markdown
# 生活百科

## 咖啡冲煮
手冲咖啡时，水温最好控制在九十摄氏度上下，研磨度要中等偏细，焖蒸三十秒后再缓慢注水，这样能充分萃取出咖啡的香气。

## 登山安全
登山前要查看天气预报，携带足够的饮水与干粮，在海拔较高的地方要注意预防高原反应，出现头晕时应立即下撤。

## 理财入门
理财的第一步是建立应急金，通常建议预留三到六个月的生活开支；在此基础上再考虑基金定投等中低风险的长期投资。

## 睡眠健康
成年人每晚应保证七到九小时的睡眠，睡前一小时避免使用发光的电子屏幕，保持卧室黑暗安静有助于入睡。
```
用例（`pytest.mark.parametrize` 四条 query→主题映射）：
- `test_semantic_retrieval_hits_target_topic`：用**换词改写**的同义 query 检索（`咖啡怎么冲泡才香？→咖啡冲煮`、`去爬山要注意什么？→登山安全`、`怎么开始攒钱投资？→理财入门`、`晚上睡不好怎么办？→睡眠健康`），`search_books(query, search_all_languages=True, top_k=1)`，断言 `top-1` 结果的 `section` 含目标主题关键字。
- `test_retrieved_content_is_verbatim_from_source`：top-k 结果的 `content` 逐行来自 `生活百科.md` 原文（沿 test_mcp_server 的逐行比对法）。
- `test_meta_records_actual_model_name`：`meta.model_name == "BAAI/bge-large-zh-v1.5"`、`meta.language == "zh"`。

### 4. `tests/test_effect_multilingual.py`（新建，集成层，真实 `bge-large-zh-v1.5` + `bge-m3`）
`@pytest.mark.integration` + 对两个模型都 `_require_model`。建一本中文书 + 一本日文书
（日文如「これは山登りと健康に関する日本語のテスト文章です。」，分段为多 section）：
- `test_non_zh_book_uses_multilingual_model`：日文书 `meta.model_name == "BAAI/bge-m3"`。
- `test_default_filters_by_query_language`：长中文 query 默认只搜中文书（所有结果 `source` 为中文书，日文书不出现）。
- `test_search_all_languages_returns_both`：`search_all_languages=True` 时结果 `source` 集合同时含中文书与日文书。

### 5. `tests/` 内新增共享 gate helper（可放 `tests/conftest.py` 顶部，或各集成文件局部定义）
```python
from huggingface_hub import try_to_load_from_cache
def require_model(repo_id: str) -> None:
    if try_to_load_from_cache(repo_id, "config.json") is None:
        pytest.skip(f"模型 {repo_id} 未缓存，跳过集成测试（先跑一次 book-rag-mcp index 即可缓存）")
```
（若 `try_to_load_from_cache` 在当前 huggingface_hub 版本不可用，回退为检查
`~/.cache/huggingface/hub/models--<org>--<name>/snapshots/` 是否存在非空目录。）

## 验收标准（全部可运行）

1. 默认快跑：`cd F:/code/projects/Book-RAG && export PYTHONPATH= && .venv/Scripts/python.exe -m pytest -q`
   全部通过（xfail 计为通过），**不联网、不下载、不加载真实模型**，耗时与现状相当（≤60s）。
2. 快速层单独跑：`.venv/Scripts/python.exe -m pytest -q tests/test_effect_language_routing.py` 通过（含 xfail 用例）。
3. 集成层单独跑：`.venv/Scripts/python.exe -m pytest -q -m integration` 通过（真实模型已在缓存，离线加载）。
4. 集成层观察项：`-m integration` 的 summary 中 `test_semantic_retrieval_hits_target_topic` 4 条全 PASS。
5. 现有 52 测试不回退：`.venv/Scripts/python.exe -m pytest -q` 总通过数 ≥ 原 52。

## 明确的 Do NOT 清单

- **不要**改任何 `src/` 源码文件（这是纯测试 + 测试配置任务，缺陷修复另立 ticket）。
- **不要**修改 `tests/test_*.py` 现有文件（FakeHashEmbeddings 在新文件内局部复制，别去重构旧的）。
- **不要**让测试触发 HuggingFace 下载：集成测试必须先 `require_model` gate，未缓存即 skip。
- **不要**在语义测试里「顺手修」短中文误判 bug（已由先决 ticket 修复）；这里用 `search_all_languages=True` 绕开即可。
- **不要**新增运行时依赖到 `pyproject.toml` 的 `dependencies`（只加 pytest 的 `markers`/`addopts` 配置）。
- **不要**提交二进制/模型/大 fixture；所有语料在 `tmp_path` 内程序化生成。
- **不要**用 `python` 直接跑（会被 Hermes 的 PYTHONPATH 污染）；一律 `.venv/Scripts/python.exe` + `export PYTHONPATH=`。

---

## 附：已确认缺陷 → 已另立先决 ticket

- **缺陷**：`search_books` 对短中文 query 返回空（`retriever.py:95-101` 语言过滤 + langdetect 短中文误判为 `ko`）。
- **处置**：修复已单独立项为 `docs/plans/book-rag-fix-short-query-langdetect.md`（CJK 脚本启发式 + TDD），
  须在**本任务书（效果测试）之前**完成并验证。本任务书快速层的短中文用例依赖该修复，落地后为普通断言。