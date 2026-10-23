"""本地 Web 服务：书籍库管理与语义搜索的浏览器入口。

设计要点（见 docs/plans/book-rag-web-ui-impl-brief.md Design 一节）：
- 全局配置沿用 ``mcp_server.py`` 的惰性初始化模式（模块级 ``_CONFIG`` +
  ``get_config()``），``create_web_app`` 可注入配置便于测试；
- **复用内核、不改内核**：检索 / 列举 / 刷新 / 书籍发现全部调用
  ``rag.retriever`` / ``rag.index`` / ``io.loader`` 的既有函数；
- 检索是同步 FAISS 调用，用 ``run_in_threadpool`` 丢进线程池避免阻塞事件循环；
- 「转换 / 更新全部」走后台线程 + 状态轮询：``refresh_all_indexes`` 同步阻塞
  且要跑 embedding，不能卡死 POST 请求。
"""

from __future__ import annotations

import dataclasses
import json
import threading
import uuid
from pathlib import Path

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route

from book_rag.config import BookRagConfig
from book_rag.io.loader import discover_books
from book_rag.rag.index import _book_key, refresh_all_indexes
from book_rag.rag.metadata import UNCLASSIFIED_WORKSPACE, load_catalog
from book_rag.rag.retriever import (
    _list_indexed_metas,
    _matches_genre,
    _matches_workspace,
    search_books,
)

# 模块级全局配置，由 get_config() 惰性初始化（create_web_app 可注入覆盖）
_CONFIG: BookRagConfig | None = None

# 后台刷新任务状态：job_id -> {"status": "running"} |
# {"status": "done", "result": {...}} | {"status": "error", "message": str}
_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()

#: BookIndexMeta 对外暴露的字段（不暴露内部 book_hash）
_META_FIELDS = (
    "book_path",
    "language",
    "chunk_count",
    "model_name",
    "updated_at",
    "workspaces",
    "genre",
    "tags",
)

#: 前端单文件页面路径（servers/ 的父目录是 book_rag/，其下 web/index.html）
_INDEX_HTML_PATH = Path(__file__).resolve().parents[1] / "web" / "index.html"


def get_config() -> BookRagConfig:
    """返回全局配置；首次调用时从环境变量构建。"""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = BookRagConfig.from_env()
    return _CONFIG


# ---------------------------------------------------------------------------
# 后台刷新任务
# ---------------------------------------------------------------------------


def _run_refresh_job(job_id: str, config: BookRagConfig) -> None:
    """在工作线程中执行增量同步，并把结果写回 _JOBS。"""
    try:
        result = refresh_all_indexes(config)
        with _JOBS_LOCK:
            _JOBS[job_id] = {"status": "done", "result": dataclasses.asdict(result)}
    except Exception as exc:  # noqa: BLE001 - 后台任务兜底，把异常转为可轮询状态
        with _JOBS_LOCK:
            _JOBS[job_id] = {"status": "error", "message": str(exc)}


# ---------------------------------------------------------------------------
# 端点处理函数
# ---------------------------------------------------------------------------


async def homepage(request: Request) -> HTMLResponse | JSONResponse:
    """返回前端单文件页面；文件缺失时返回 500 与明确错误文案。"""
    if not _INDEX_HTML_PATH.is_file():
        return JSONResponse(
            {"detail": f"前端页面缺失：{_INDEX_HTML_PATH}（请确认安装完整）"},
            status_code=500,
        )
    return HTMLResponse(_INDEX_HTML_PATH.read_text(encoding="utf-8"))


async def api_books(request: Request) -> JSONResponse:
    """已索引书籍列表：BookIndexMeta 的对外字段投影。

    支持 ``?workspace=<id>&genre=<genre>`` 过滤；元数据按 workspaces.yaml 实时解析。
    """
    config = get_config()
    workspace = request.query_params.get("workspace") or None
    genre = request.query_params.get("genre") or None
    metas = _list_indexed_metas(config.index_dir, config.library_path)
    if workspace:
        metas = [meta for meta in metas if _matches_workspace(meta, workspace)]
    if genre:
        metas = [meta for meta in metas if _matches_genre(meta, genre)]
    return JSONResponse(
        [{field: getattr(meta, field) for field in _META_FIELDS} for meta in metas]
    )


async def api_workspaces(request: Request) -> JSONResponse:
    """返回 workspace 定义与已索引书籍数量，供前端切换下拉使用。

    未在 yaml 中定义但实际出现的 workspace（典型为 ``_unclassified``）也会列出。
    """
    config = get_config()
    catalog = load_catalog(config.library_path)
    metas = _list_indexed_metas(config.index_dir, config.library_path)

    counts: dict[str, int] = {}
    for meta in metas:
        for workspace in meta.workspaces:
            counts[workspace] = counts.get(workspace, 0) + 1

    items = [
        {
            "id": workspace.id,
            "label": workspace.label,
            "genre": workspace.genre,
            "book_count": counts.get(workspace.id, 0),
        }
        for workspace in catalog.workspaces.values()
    ]
    for workspace_id in sorted(counts):
        if workspace_id in catalog.workspaces:
            continue
        items.append(
            {
                "id": workspace_id,
                "label": "未分类" if workspace_id == UNCLASSIFIED_WORKSPACE else workspace_id,
                "genre": "unknown",
                "book_count": counts[workspace_id],
            }
        )
    # _unclassified 始终排在最后，避免未声明书混进正常分组
    items.sort(key=lambda item: item["id"] == UNCLASSIFIED_WORKSPACE)
    return JSONResponse(items)


async def api_books_available(request: Request) -> JSONResponse:
    """库内发现的书籍列表，逐本标注是否已有索引（以 manifest 为准）。"""
    config = get_config()
    manifest_path = config.index_dir / "manifest.json"
    indexed_keys: set[str] = set()
    if manifest_path.is_file():
        indexed_keys = set(json.loads(manifest_path.read_text(encoding="utf-8")))

    items = []
    for book in discover_books(config.library_path):
        key = _book_key(book, config.library_path).as_posix()
        items.append({"path": key, "indexed": key in indexed_keys})
    return JSONResponse(items)


async def api_search(request: Request) -> JSONResponse:
    """语义搜索：body ``{query, top_k=5, book=null, search_all_languages=false,
    workspace=null, genre=null}``。"""
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 - 非法 JSON 统一按客户端错误处理
        return JSONResponse({"detail": "请求体不是合法 JSON"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"detail": "请求体应为 JSON 对象"}, status_code=400)

    query = body.get("query")
    if not isinstance(query, str) or not query.strip():
        return JSONResponse({"detail": "query 不能为空"}, status_code=400)

    try:
        top_k = int(body.get("top_k", 5))
    except (TypeError, ValueError):  # 非整数不报 400，退回默认值
        top_k = 5
    top_k = max(1, min(50, top_k))

    book = body.get("book") or None
    books = body.get("books")
    group_by_book = body.get("group_by_book", False)
    if book is not None and not isinstance(book, str):
        return JSONResponse({"detail": "book 必须为字符串"}, status_code=400)
    if not isinstance(group_by_book, bool):
        return JSONResponse({"detail": "group_by_book 必须为布尔值"}, status_code=400)
    search_all_languages = bool(body.get("search_all_languages", False))
    workspace = body.get("workspace") or None
    if not isinstance(workspace, (str, list)):
        workspace = None
    genre = body.get("genre") or None
    if not isinstance(genre, str):
        genre = None

    # 同步 FAISS 搜索丢线程池，避免阻塞 async 事件循环
    try:
        results = await run_in_threadpool(
            search_books, query, config=get_config(), top_k=top_k, book=book, books=books,
            search_all_languages=search_all_languages, workspace=workspace, genre=genre,
        )
    except ValueError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=400)
    items = [dataclasses.asdict(r) for r in results]
    # FAISS 距离参与运算后 score 为 numpy.float32，JSON 序列化前转原生类型
    for item in items:
        item["score"] = float(item["score"])
        item["position"] = int(item["position"])
    response = {"results": items}
    if group_by_book:
        groups = {}
        for item in items:
            groups.setdefault(item["source"], []).append(item)
        response["groups"] = [{"source": source, "results": group}
                              for source, group in groups.items()]
    return JSONResponse(response)


async def api_refresh(request: Request) -> JSONResponse:
    """提交一次后台增量同步任务；已有 running 任务时幂等返回其 job_id。"""
    with _JOBS_LOCK:
        for existing_id, job in _JOBS.items():
            if job["status"] == "running":
                return JSONResponse({"job_id": existing_id})

        job_id = uuid.uuid4().hex
        _JOBS[job_id] = {"status": "running"}

    thread = threading.Thread(
        target=_run_refresh_job, args=(job_id, get_config()), daemon=True
    )
    thread.start()
    return JSONResponse({"job_id": job_id})


async def api_refresh_status(request: Request) -> JSONResponse:
    """查询刷新任务状态：running / done（含统计）/ error（含 message）。"""
    job_id = request.path_params["job_id"]
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        return JSONResponse({"detail": f"未知的刷新任务：{job_id}"}, status_code=404)
    return JSONResponse(job)


async def _unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """兜底异常处理：统一返回 500 JSON。"""
    return JSONResponse({"detail": f"服务器内部错误：{exc}"}, status_code=500)


def create_web_app(config: BookRagConfig | None = None) -> Starlette:
    """创建 Web 应用实例；传入 config 时同时更新模块级全局配置。"""
    global _CONFIG
    if config is not None:
        _CONFIG = config

    return Starlette(
        routes=[
            Route("/", homepage, methods=["GET"]),
            Route("/api/books", api_books, methods=["GET"]),
            Route("/api/books/available", api_books_available, methods=["GET"]),
            Route("/api/workspaces", api_workspaces, methods=["GET"]),
            Route("/api/search", api_search, methods=["POST"]),
            Route("/api/refresh", api_refresh, methods=["POST"]),
            Route("/api/refresh/{job_id}", api_refresh_status, methods=["GET"]),
        ],
        exception_handlers={Exception: _unhandled_error_handler},
    )
