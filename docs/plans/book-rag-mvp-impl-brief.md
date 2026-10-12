# Book-RAG MCP MVP 实现任务书

> **背景**：把 Book-RAG 从当前占位骨架实现为可用的「书籍知识检索 MCP 服务」。用户核心需求：整理本地书籍 → 自动索引 → 通过 MCP 工具按书的内容检索并返回原文片段。
>
> **技术约束**：
> - 使用本地 HuggingFace embedding（隐私、离线、零成本）。
> - 按书籍语言自动选择专用 embedding 模型。
> - 支持 PDF / EPUB / TXT / Markdown。
> - 索引持久化，启动自动加载，自动检测目录变化并增量更新。
> - 通过 MCP 暴露工具给 Hermes / Claude Code / Cursor 等 Agent 使用。
> - 只返回书中相关原文片段，不生成回答。

## 架构总览

```
AI Agent (Hermes) ──MCP stdio/http──► FastMCP Server
                                            │
        ┌───────────────────────────────────┼──────────────────────────────┐
        ▼                                   ▼                              ▼
  search_book_knowledge                list_indexed_books            refresh_book_index
        │                                       │                              │
        ▼                                       │                              ▼
  detect_query_language                         │                       scan library
        │                                       │                       compare hash/mtime
        ▼                                       ▼                              │
  for each matching book index                  list metadata                rebuild changed
  ──► encode query with book's model                                         delete removed
      search FAISS
      merge & rerank by score
      return snippets + metadata
```

### 关键设计决策

1. **每本书独立 FAISS 索引 + 元数据 JSON**  
   全局索引难以做增量删除，按书隔离后：新增/修改/删除一本书只需重建/删除该书索引，不影响其他书。

2. **按语言选专用 embedding 模型**  
   索引时检测书籍主导语言，中文用 `BAAI/bge-large-zh-v1.5`，英文用 `BAAI/bge-large-en-v1.5`，其他/混合用 `BAAI/bge-m3`。每本书记录实际使用的模型，检索时用同一模型编码 query。

3. **query 语言自动路由**  
   默认只搜索与 query 同语言的书籍，避免跨语言向量空间引入噪声；提供 `search_all_languages` 开关供跨语言检索。

4. **结构感知切分 + Parent-Doc 返回**  
   优先按自然段落/标题切分，不硬截断句子；搜索用较小片段提高精度，返回时附带所在章节路径（section path）与前后文范围，最大程度保留原意。

5. **增量索引**  
   维护全局 `manifest.json`，记录每本书的文件路径、sha256 hash、修改时间。启动服务时自动扫描目录，仅对新增或变更的书重建索引，删除已不存在的书的索引。

6. **依赖最小化**  
   本次只新增 `langdetect`；不引入 `langchain_openai` 等 API 依赖（为后续火山方舟扩展预留接口，但不在 MVP 实现）。

---

## 文件改动清单

### 1. `pyproject.toml`
- 新增依赖：`"langdetect>=1.0"`
- 确保现有依赖完整：`fastmcp>=2.0`、`langchain-text-splitters>=0.3`、`langchain-community>=0.3`、`langchain-huggingface>=0.1`、`sentence-transformers>=3.0`、`faiss-cpu>=1.8`、`pypdf>=5.0`、`ebooklib>=0.18`、`beautifulsoup4>=4.12`、`pydantic>=2.0`

### 2. `src/book_rag/config.py`（新建）
- `BookRagConfig` dataclass：
  - `library_path: Path`
  - `index_dir: Path`
  - `chunk_size: int = 512`
  - `chunk_overlap: int = 64`
  - `default_top_k: int = 5`
- `from_env()`：从环境变量 `BOOK_RAG_LIBRARY`、`BOOK_RAG_INDEX_DIR` 读取，默认 `index_dir = library_path / ".book_rag_index"`。

### 3. `src/book_rag/io/parsers.py`（新建）
- `BookSection` dataclass：`path: str`、`level: int`、`text: str`、`position: int`
- `ParsedBook` dataclass：`title: str`、`language: str | None`、`sections: list[BookSection]`
- `BookParser(ABC)` 接口：`parse(path: Path) -> ParsedBook`
- 实现：
  - `MarkdownParser`：按 `#` 标题层级切分。
  - `TextParser`：按空行分段，尝试识别类似 `第 X 章` / `Chapter X` 的标题行。
  - `PdfParser`：用 `pypdf` 逐页提取文本，按页 + 段落组织 section。
  - `EpubParser`：用 `ebooklib` + `bs4` 提取各章节文本。
- `get_parser(suffix: str) -> BookParser` 工厂函数。

### 4. `src/book_rag/io/loader.py`（新建）
- `SUPPORTED_SUFFIXES = {".pdf", ".epub", ".txt", ".md", ".markdown"}`
- `discover_books(library_path: Path) -> list[Path]`：递归查找受支持的书籍文件。
- `load_book(path: Path) -> ParsedBook`：选择 parser 解析。
- `compute_file_hash(path: Path) -> str`：SHA256。

### 5. `src/book_rag/rag/language.py`（新建）
- `detect_language(text: str) -> str`：用 `langdetect.detect`，失败返回 `"unknown"`。
- `normalize_language(lang: str) -> str`：把 `zh-cn`、`zh-tw` 等映射为 `zh`；`en-us` 等映射为 `en`。
- `choose_embedding_model(language: str) -> str`：映射表见上文。

### 6. `src/book_rag/rag/embeddings.py`（新建）
- `_embedding_cache: dict[str, HuggingFaceEmbeddings]` 按模型名缓存实例。
- `get_embeddings(model_name: str) -> HuggingFaceEmbeddings`：从缓存取或新建。
- `get_embeddings_for_language(language: str) -> HuggingFaceEmbeddings`：先 `choose_embedding_model` 再 `get_embeddings`。

### 7. `src/book_rag/rag/splitter.py`（新建）
- `split_book(sections: list[BookSection], chunk_size: int, chunk_overlap: int) -> list[Document]`：
  - 使用 `RecursiveCharacterTextSplitter`。
  - separators 优先 `\n\n`、`\n`、`。`、`．`、`. `、` `、`""`。
  - 每个 chunk 的 metadata 必须包含：`source`（相对路径）、`title`（书名）、`section`（章节路径）、`position`（全局顺序）。

### 8. `src/book_rag/rag/index.py`（新建）
- `BookIndexMeta` dataclass：`book_path`、`book_hash`、`language`、`model_name`、`chunk_count`、`updated_at`，可序列化为 JSON。
- `get_book_index_dir(index_dir: Path, book_path: Path) -> Path`：用书籍相对路径生成稳定目录名（如相对路径的 safe hash）。
- `build_book_index(book_path: Path, config: BookRagConfig) -> BookIndexMeta`：
  1. `load_book`
  2. 若 `parsed.language` 为空则采样检测语言
  3. `split_book`
  4. `FAISS.from_documents(docs, get_embeddings_for_language(language))`
  5. `save_local`
  6. 写 `meta.json`
- `load_book_index(index_dir: Path, book_path: Path) -> tuple[FAISS, BookIndexMeta] | None`：加载索引与元数据，返回 None 表示不存在。
- `refresh_all_indexes(config: BookRagConfig) -> RefreshResult`：扫描目录、对比 manifest、重建变更、删除失效，返回统计信息。

### 9. `src/book_rag/rag/retriever.py`（新建）
- `SearchResult` dataclass：`content: str`、`book: str`、`section: str`、`source: str`、`score: float`、`position: int`
- `search_books(query: str, *, config: BookRagConfig, top_k: int = 5, book: str | None = None, search_all_languages: bool = False) -> list[SearchResult]`：
  1. 检测 query 语言。
  2. 加载所有书籍索引。
  3. 按 `book` 过滤（支持部分匹配）。
  4. 默认按 query 语言过滤书籍；`search_all_languages=True` 时搜索全部。
  5. 对每本匹配的书，用其模型编码 query，执行 `similarity_search_with_score`，结果归一化后合并。
  6. 按 score 排序返回前 `top_k`。

### 10. `src/book_rag/servers/mcp_server.py`（新建）
- 初始化全局 `config = BookRagConfig.from_env()`。
- 启动时调用 `refresh_all_indexes(config)` 做一次增量同步。
- FastMCP 工具：
  - `search_book_knowledge(query: str, top_k: int = 5, book: str | None = None, search_all_languages: bool = False) -> str`：
    调用 `search_books`，把结果格式化为 Markdown 文本返回（包含书名、章节、相关度、原文片段）。
  - `list_indexed_books() -> list[str]`：返回 `"书名 (语言: zh, 块数: 123)"` 列表。
  - `refresh_book_index() -> str`：调用 `refresh_all_indexes`，返回增量同步统计文本。

### 11. `src/book_rag/cli.py`（重写）
- 保留 `--transport {stdio,http}`。
- `serve` 子命令（默认）：启动 MCP server。
- `index` 子命令：构建/刷新索引后退出。
- 移除占位打印。

### 12. `tests/`（新建）
按 TDD 每个模块对应一个测试文件：
- `tests/test_config.py`
- `tests/test_parsers.py`（用 fixtures 放最小 PDF/EPUB/TXT/MD）
- `tests/test_language.py`
- `tests/test_splitter.py`
- `tests/test_index.py`
- `tests/test_retriever.py`
- `tests/test_mcp_server.py`（mock 索引，避免加载真实模型）

### 13. `README.md`（更新）
- 更新「当前状态」为「MVP 实现中 / 已实现」。
- 更新项目结构标记。
- 补充 Hermes `mcp_servers` 配置示例。

---

## 增量索引规则（精确行为）

```python
manifest[book_path] = {"hash": sha256, "mtime": mtime}

for each discovered book:
    if book not in manifest:
        build_book_index(book)          # 新书
    elif manifest[book].hash != current_hash:
        build_book_index(book)          # 已变更
    else:
        skip                            # 未变更

for each book in manifest but not in discovered books:
    delete its index directory           # 已删除

write manifest
```

---

## 验收标准（必须全部可运行）

1. **语法/导入检查**
   - `python -m py_compile src/book_rag/**/*.py`
   - `python -c "import book_rag"`

2. **单元测试**
   - `python -m pytest tests/ -v` 全部通过。

3. **解析器测试**
   - 对 fixtures 中的 TXT / MD / PDF / EPUB 都能提取出非空文本与至少一个 section。

4. **索引端到端**
   - 在 `BOOK_RAG_LIBRARY` 下放置几本测试书，运行 `uv run book-rag-mcp index` 后：
     - `.book_rag_index/manifest.json` 存在且条目正确。
     - 每本书对应索引目录存在 `index.faiss` 与 `meta.json`。

5. **检索端到端**
   - 索引建立后，运行 Python 脚本调用 `search_books("某个书中概念", config=config)`：
     - 返回结果非空。
     - 每个结果包含 `content`、`book`、`section`、`score`。
     - 结果内容确实来自对应书籍。

6. **MCP 冒烟**
   - 配置 Hermes `mcp_servers` 使用 `uv --directory F:/code/projects/Book-RAG run book-rag-mcp`。
   - Hermes 启动后能发现工具 `mcp_book_rag_search_book_knowledge`。
   - 调用该工具能返回格式化原文片段。

7. **增量索引**
   - 新增一本书后重新启动服务，能自动为新书建立索引而不重建旧书。
   - 修改一本书后重新启动，能自动重建该书索引。
   - 删除一本书后，其索引目录被清理。

---

## 明确的 Do NOT 清单

- **不要**在 MVP 中接入火山方舟或任何在线 embedding API（只保留扩展接口，不实现）。
- **不要**实现聊天式回答生成，MCP 工具只返回原文片段。
- **不要**实现 Web UI 或 REST API（只通过 MCP 暴露）。
- **不要**改动 `uv.lock` 以外的 lock 文件；依赖变更通过 `pyproject.toml` 后由 `uv sync` 更新。
- **不要**把测试 fixture 中的真实/大文件提交到仓库；fixture 应小于 50KB。
- **不要**在源码中硬编码书籍库路径或模型路径；全部走 `BookRagConfig`。
- **不要**让测试下载真实 HuggingFace 模型；涉及 embedding 的测试必须 mock `get_embeddings`。

---

## 分块执行建议（交给 Claude Code）

| 阶段 | 范围 | 目标 |
|---|---|---|
| **Pilot** | `config.py` + `language.py` + `parsers.py` + 对应测试 | 验证解析与配置路径 |
| **Chunk 1** | `splitter.py` + `embeddings.py` + `loader.py` + 测试 | RAG 基础组件 |
| **Chunk 2** | `index.py` + `retriever.py` + 测试 | 索引与检索核心 |
| **Chunk 3** | `mcp_server.py` + `cli.py` + 集成测试 + README 更新 | MCP 接入与交付 |

每个阶段必须：先写失败测试 → 最小实现 → 测试通过 → commit，再进入下一阶段。
