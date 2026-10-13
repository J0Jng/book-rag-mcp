# Book-RAG 修复：短中文查询被 langdetect 误判致检索返回空

> **背景**：架构师实测发现，`search_books` 对短中文 query 返回空结果。根因在
> `src/book_rag/rag/language.py` 的 `detect_language`：它直接调用 `langdetect.detect`，
> 而 langdetect 对**短中文文本**误判率高——「招聘」「咖啡怎么冲泡才香？」被判成韩语 `ko`。
> 随后 `src/book_rag/rag/retriever.py:95-101` 用这个结果做「只搜同语言书籍」过滤，
> 把语言为 `zh` 的中文书全部排除，返回空。
>
> **实测证据**（真实索引 + 真实查询）：
> - `咖啡怎么冲泡才香？` → langdetect `ko` → hits=0
> - `招聘` → `ko` → hits=0
> - `人生设计` / `怎么读书` / `如何找到适合自己的职业方向` → `zh` → hits=3

## 设计决策

**用 CJK 脚本启发式优先、langdetect 兜底**（不引入新依赖，不换检测库）：

- 文本含**假名**（U+3040–U+30FF）→ `ja`（日语几乎必然带假名，中文/韩语不含）
- 否则含**谚文**（U+AC00–U+D7AF、U+1100–U+11FF）→ `ko`
- 否则含**汉字**（U+4E00–U+9FFF）→ `zh`（纯汉字文本几乎必是中文）
- 否则返回 `None`，回退到 `langdetect.detect`（处理拉丁/西里尔等非 CJK 文本）

理由：脚本判定对 CJK 是确定性的、零成本、离线；langdetect 仍兜底非 CJK 语言。
该修复同时惠及**索引侧**（`index.py:_sample_language` 检测书籍语言）与**检索侧**
（query 语言），一处修复两处生效。

## 改动清单

### 1. `src/book_rag/rag/language.py`
- 顶部新增 `import re`（若未导入）。
- 新增三个模块级编译正则（放在 `_EMBEDDING_MODEL_MAP` 附近）：
  - `_HANGUL_RE = re.compile(r"[\uac00-\ud7af\u1100-\u11ff]")`
  - `_KANA_RE = re.compile(r"[\u3040-\u30ff]")`
  - `_HAN_RE = re.compile(r"[\u4e00-\u9fff]")`
- 新增私有函数 `_detect_cjk_script(text: str) -> str | None`：按「kana→ja、hangul→ko、han→zh、否则 None」判定。
- 修改 `detect_language`：在空文本检查之后、调用 `detect` 之前，先 `cjk = _detect_cjk_script(text); if cjk: return cjk`，否则走原有 `langdetect.detect` + `LangDetectException` 兜底。
- 现有公开函数签名与返回契约不变（仍返回 `"unknown"` 表示无法判定）。

### 2. `tests/test_language.py`（TDD：先写失败测试再实现）
新增用例（当前实现下这些会 RED，修复后 GREEN）：
- `test_detect_language_short_chinese_word`：`detect_language("招聘") == "zh"`。
- `test_detect_language_short_chinese_question`：`normalize_language(detect_language("咖啡怎么冲泡才香？")) == "zh"`。
- `test_detect_language_japanese_kana`：`detect_language("これは日本語のテストです") == "ja"`。
- `test_detect_language_korean_hangul`：`detect_language("안녕하세요 반갑습니다") == "ko"`。
- 回归（确保非 CJK 仍走 langdetect）：已有 `test_detect_language_english` / `test_detect_language_chinese` / `test_detect_language_empty_returns_unknown` 不得破坏。

## 验收标准（全部可运行）

1. 快速层单测：`cd F:/code/projects/Book-RAG && export PYTHONPATH= && .venv/Scripts/python.exe -m pytest tests/test_language.py -q` 全绿。
2. 全量回归：`.venv/Scripts/python.exe -m pytest -q` 全绿（原 52 通过 + 新增 4 通过）。
3. 真实冒烟（架构师亲自复核）：用真实索引跑以下脚本，两条短中文 query 均 `hits>0`：
   ```python
   from book_rag.config import BookRagConfig
   from book_rag.rag.retriever import search_books
   cfg = BookRagConfig.from_env()
   for q in ["招聘", "咖啡怎么冲泡才香？"]:
       print(q, len(search_books(q, config=cfg, top_k=3)))
   ```
4. `from book_rag.rag.language import detect_language` 导入无报错。

## 明确的 Do NOT 清单

- **不要**引入新依赖（不装 `lingua` / `fasttext` / `langid`），只加 `re` + 字符区间正则。
- **不要**改 `search_books` / `retriever.py` / `index.py` / `normalize_language` —— 修复仅限 `language.py` 的 `detect_language` 及其私有 helper。
- **不要**改动 `test_language.py` 之外的任何测试文件。
- **不要**改 `pyproject.toml`（无依赖变更）。
- **不要**用 `python` 直接跑（PYTHONPATH 污染）；一律 `.venv/Scripts/python.exe` + `export PYTHONPATH=`。
- 保持 `DetectorFactory.seed = 0` 与 `detect_language` 的「空文本返回 unknown」语义不变。