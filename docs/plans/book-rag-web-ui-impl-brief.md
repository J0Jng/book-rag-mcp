# Implementation Brief: Book-RAG 本地 Web 前端（批量转换 + 数据搜索查看）

## Background

Book-RAG 目前已有完整 RAG 内核（解析 → 切分 → 本地 embedding → FAISS 索引）与两个入口：
CLI（`book-rag-mcp index` / `serve`）和 MCP 服务。但使用者（Joe 本人）要**不借助任何 Agent**
就能批量把书籍转换入库、以及做语义搜索查看，目前只能靠命令行，没有图形界面。

本任务的产物：一个本地 Web 服务 + 单文件前端页面，直接复用现有 RAG 内核函数，
不重复实现任何解析/切分/向量化/检索逻辑。

已与 Joe 确认两项决策：
1. 后端用 **Starlette + uvicorn**（环境中已随 fastmcp 引入 starlette 1.6.0、uvicorn 0.52.4，
   只需在 `pyproject.toml` 显式声明依赖，零新增重依赖）。
2. 「批量转换」= **后台线程 + 状态轮询**（`refresh_all_indexes` 同步阻塞且要跑 embedding，
   不能卡死 POST 请求）。

## Design

### 后端（`src/book_rag/servers/web.py`）

- `create_web_app(config: BookRagConfig | None = None) -> Starlette`：
  config 沿用 `mcp_server.py` 的惰性初始化模式（模块级 `_CONFIG` + `get_config()`），
  `None` 时 `BookRagConfig.from_env()`。
- **复用内核、不改内核**：端点直接调用现成函数——
  - 已索引书籍：`book_rag.rag.retriever._list_indexed_metas(config.index_dir)` → `BookIndexMeta`
  - 库内书籍发现：`book_rag.io.loader.discover_books(config.library_path)`
  - 检索：`book_rag.rag.retriever.search_books(query, config=..., top_k=..., book=..., search_all_languages=...)`
  - 增量转换：`book_rag.rag.index.refresh_all_indexes(config)` → `RefreshResult`
- **序列化**：`dataclasses.asdict`。`BookIndexMeta` 只投影对外字段
  `book_path / language / chunk_count / model_name / updated_at`（不暴露内部 `book_hash`）。
  `SearchResult` 全字段序列化（`content/book/section/source/score/position`）。
- **阻塞调用丢线程池**：检索端点用 `starlette.concurrency.run_in_threadpool` 包裹 `search_books`，
  避免同步 FAISS 搜索阻塞 async 事件循环。
- **后台 job**：模块级 `_JOBS: dict[str, dict]`；`POST /api/refresh` 生成 `uuid4().hex` 作为 job_id，
  用 `threading.Thread(target=..., daemon=True)` 执行 `refresh_all_indexes`，把 `RefreshResult`
  转为 dict 存进 `_JOBS[job_id]`（状态 `running` → `done`；异常 → `error` + `message`）。
  若已有 `running` 状态的 job 则返回其 job_id（幂等，避免重复并发建索引）。
- `GET /` 返回 `index.html`：用 `Path(__file__).resolve().parents[1] / "web" / "index.html"`
  读取（`servers/` 的父目录是 `book_rag/`，其下 `web/index.html` 为前端文件）。
  读不到时返回 500 并给出明确错误文案。

### 端点

| 方法 | 路径 | 行为 |
|---|---|---|
| GET | `/` | 返回 `index.html` |
| GET | `/api/books` | 已索引书籍列表 `[{book_path, language, chunk_count, model_name, updated_at}]` |
| GET | `/api/books/available` | 库内发现的书籍 `[{path, indexed}]`（`indexed` 由 manifest 是否含该书判定） |
| POST | `/api/search` | body `{query, top_k=5, book=null, search_all_languages=false}` → `{results:[...]}` |
| POST | `/api/refresh` | `→ {job_id}` |
| GET | `/api/refresh/{job_id}` | `→ {status:"running"|"done"|"error", result?|message?}` |

参数校验/错误处理：
- `query` 缺失或空白 → 400 `{"detail": "query 不能为空"}`；
- `top_k` 非整数或越界 → clamp 到 `[1, 50]`（不报 400，宽容处理）；
- 未知 job_id → 404；异常统一 → 500 JSON（含 message）。

### CLI（`src/book_rag/cli.py`）

在 `build_parser()` 新增 `--host`（默认 `127.0.0.1`）、`--port`（默认 `8000`）两个可选参数，
并在 `action` 的 `choices` 中新增 `"web"`。`main()` 中新增分支：
`web` 时 `uvicorn.run(create_web_app(config), host=args.host, port=args.port)`。
`serve`（MCP）与 `index` 行为保持不变。

### HF embedding 首次加载加速（重要坑）

`huggingface_hub` 在**首次 import 时**固化 `HF_HUB_OFFLINE` 等 env。web 服务经 CLI 启动
**不经过** `tests/conftest.py`，未设该 env 时首次加载 embedding 会触发未认证的 Hub 网络校验，
耗时从 ~16s 拖到 ~134s。处理：在 `cli.py` **模块最顶部、所有 import 之前**加
（与 `conftest.py` 同款 `setdefault`，不覆盖用户显式设置）：

```python
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
```

`book_rag/__init__.py` 只有 docstring + `__version__`（不 import 重模块），因此放在
`cli.py` 顶部即早于 `book_rag.rag.embeddings` 的首次 import，`serve` 与 `web` 都受益。

### 前端（`src/book_rag/web/index.html`）

单文件，内嵌 `<style>` 与 `<script>`，原生 `fetch`，**不引入任何外部 CDN / 框架 / 构建链**。
系统字体栈；浅色简洁风格；文案中文。布局自上而下三个区块 + 页头：

1. **页头**：标题「Book-RAG 资料库」+ 一句说明。
2. **📚 书籍库 / 批量转换**：
   - 加载时 `GET /api/books/available`，列出库内每本书（`path` + 「已索引 / 未索引」badge）。
   - 「🔄 转换 / 更新全部」按钮：`POST /api/refresh` → 拿到 `job_id` → 每 1s 轮询
     `GET /api/refresh/{job_id}`，显示运行中状态；`done` 后展示统计
     （新建 / 跳过 / 删除 三列，各自书名列表），并刷新两个书籍列表。
2. **🔍 数据搜索**：
   - 输入：query 文本框、`top_k` 数字（1–50，默认 5）、书籍过滤（可选文本，留空=全部）、
     「跨语言搜索」checkbox。
   - 提交 `POST /api/search`；结果以卡片列出：`书名 · 章节`、相关度 `score`（显示为百分比）、
     原文片段（保留换行）。无结果时显示「未找到相关内容」。搜索中显示 loading。
3. **📖 已索引书籍**：`GET /api/books`，表格列 `书名 / 语言 / 块数 / 更新时间`；
   空列表显示引导文案「还没有索引，点上方转换按钮」。

视觉注意（避免 AI 痕迹）：不用全大写 label、不做无意义编号 eyebrow、正文用正常句读；
相关度可用一个细进度条而非只给数字。

## Change List

### 1. `pyproject.toml`
- 在 `[project].dependencies` 列表追加两行（**不要 touch 任何已有依赖行，不要改版本号**）：
  `"starlette>=0.40",`、`"uvicorn>=0.30",`
- 运行 `uv sync` 更新锁文件（只应影响新增依赖的解析，若发现大范围 diff 立即停下报告）。
- hatchling 默认只打包 `.py`：在 `[tool.hatch.build.targets.wheel]` 下新增 `[tool.hatch.build.targets.wheel.force-include]`，
  `"src/book_rag/web/index.html" = "book_rag/web/index.html"` 使前端随 wheel 分发
  （本地 `uv run` 直接源码可跑，打包可选但应配好）。

### 2. `src/book_rag/cli.py`（新增）
- 模块顶部（`import argparse` 之前）加 `import os` + 两行 `setdefault`（见上）。
- `build_parser()`：加 `--host`、`--port`；`action` choices 加 `"web"`。
- `main()`：加 `web` 分支，`from book_rag.servers.web import create_web_app` + `import uvicorn`，
  `uvicorn.run(create_web_app(config), host=args.host, port=args.port)`。

### 3. `src/book_rag/servers/web.py`（新增）
按上文 Design 实现：`get_config()` / `create_web_app()` / 各端点处理函数 /
后台 job 字典与线程。所有检索与刷新只调用内核既有函数，**不得复制其逻辑**。

### 4. `src/book_rag/web/index.html`（新增）
按上文「前端」实现单文件页面。

### 5. `tests/test_web.py`（新增）
- 复用 `tests/test_retriever.py` / `test_mcp_server.py` 的 `FakeHashEmbeddings` 与
  `monkeypatch`（patch `index_mod.get_embeddings` / `get_embeddings_for_language` /
  `detect_language` + `retriever_mod.detect_language`），**不下载真实模型**。
- 用 `from starlette.testclient import TestClient`（环境中 httpx 0.28.1 已就绪，`TestClient` 可用）。
- 用例覆盖：
  - `GET /` 返回 200 且 body 含 `<html`；
  - `GET /api/books`（空索引 → `[]`；建索引后 → 长度与字段正确）；
  - `GET /api/books/available`（两本书、`indexed` 状态正确）；
  - `POST /api/search`（命中返结果、字段完整；空 query → 400；`top_k` clamp）；
  - `POST /api/refresh` + `GET /api/refresh/{id}`（轮询到 `done`，`result` 含 built/skipped/deleted 统计）。

## Acceptance Criteria

1. `export PYTHONPATH= && uv run pytest -q` 全绿（含新增 `test_web.py`；真实 embedding 集成测试按现有 marker 自动跳过）。
2. `export PYTHONPATH= && uv run book-rag-mcp web --port 8765` 后台启动后：
   - `curl -s http://127.0.0.1:8765/api/books` 返回 JSON 数组（对现有 `books/` 那本《斯坦福人生设计课》索引，应含 1 项）；
   - `curl -s -X POST http://127.0.0.1:8765/api/search -H "Content-Type: application/json" -d '{"query":"人生设计"}'`
     返回含 `results` 的 JSON，且结果 `content` 确实是该书原文片段（**真实**，非 mock）；
   - `curl -s -X POST http://127.0.0.1:8765/api/refresh` 返回 `{job_id}`，轮询其 status 最终为 `done`；
   - `curl -s http://127.0.0.1:8765/` 返回 HTML。
3. 浏览器打开 `http://127.0.0.1:8765/` 能显示页面，完成一次真实搜索与一次「转换全部」。

## Explicitly Do NOT Do

- ❌ 不改 RAG 内核任何文件（`rag/index.py` / `retriever.py` / `embeddings.py` / `splitter.py` / `language.py`、
  `io/parsers.py` / `loader.py` / `config.py`）现有逻辑。
- ❌ 不改 `servers/mcp_server.py` 现有工具与行为。
- ❌ 不引入 React / Vue / 构建工具链 / npm / 外部 CDN。
- ❌ 不加数据库、不加登录认证、不加「格式转换」（本任务「转换」= 解析建索引，不是 epub→txt）。
- ❌ 不重写 `search_books` / `refresh_all_indexes` 的实现，只能以调用者身份复用。
- ❌ 不改已存在的依赖行 / 版本号；只追加 `starlette`、`uvicorn` 两行。
- ❌ 不新增除 `starlette`、`uvicorn` 外的任何依赖（测试用库 httpx 已具备）。
- ❌ 不提交 commit、不 push（由 Hermes 验收后统一处理）。