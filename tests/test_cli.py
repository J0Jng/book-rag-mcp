"""cli 模块测试。

覆盖行为：
- ``index`` 子命令：正确调用 ``refresh_all_indexes``（传入 from_env 配置），
  打印同步统计后以 0 退出；
- ``serve`` 子命令（默认）：实例化 MCP server 并以指定 transport 启动；
- ``--transport http`` 生效。

serve 路径的真实启动（阻塞运行事件循环）不在单测范围，仅验证装配逻辑；
create_mcp 本身的行为见 test_mcp_server.py。
"""

import shutil
from pathlib import Path

import pytest

from book_rag import cli
from book_rag.config import BookRagConfig
from book_rag.rag.index import RefreshResult

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def env_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> BookRagConfig:
    """用环境变量把书籍库 / 索引目录指向临时路径，模拟 from_env 的真实行为。"""
    library = tmp_path / "library"
    library.mkdir()
    shutil.copy(FIXTURES / "sample_book.md", library / "sample_book.md")
    monkeypatch.setenv("BOOK_RAG_LIBRARY", str(library))
    monkeypatch.delenv("BOOK_RAG_INDEX_DIR", raising=False)
    return BookRagConfig(library_path=library, index_dir=library / ".book_rag_index")


# ---------------------------------------------------------------------------
# index 子命令
# ---------------------------------------------------------------------------


def test_index_subcommand_refreshes_and_exits(
    env_config: BookRagConfig, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    """index 子命令调用一次 refresh，配置来自 from_env，打印统计后退出 0。"""
    calls: list[BookRagConfig] = []
    monkeypatch.setattr(
        cli,
        "refresh_all_indexes",
        lambda config: calls.append(config) or RefreshResult(built=["sample_book.md"]),
    )

    code = cli.main(["index"])

    assert code == 0
    assert len(calls) == 1
    assert isinstance(calls[0], BookRagConfig)
    assert calls[0].library_path == env_config.library_path
    assert calls[0].index_dir == env_config.index_dir

    out = capsys.readouterr().out
    assert "索引同步完成" in out
    assert "sample_book.md" in out


# ---------------------------------------------------------------------------
# serve 子命令
# ---------------------------------------------------------------------------


class _StubMcp:
    """记录 run() 调用参数的 MCP server 桩。"""

    def __init__(self):
        self.run_calls: list[str] = []

    def run(self, transport: str) -> None:
        self.run_calls.append(transport)


def test_serve_subcommand_builds_server_with_env_config(
    env_config: BookRagConfig, monkeypatch: pytest.MonkeyPatch
):
    """serve 子命令用 from_env 的配置实例化 MCP server，并以 stdio 启动。"""
    created: list[BookRagConfig] = []
    stub = _StubMcp()
    monkeypatch.setattr(
        cli, "create_mcp", lambda config: (created.append(config), stub)[1]
    )

    code = cli.main(["serve"])

    assert code == 0
    assert len(created) == 1
    assert isinstance(created[0], BookRagConfig)
    assert created[0].library_path == env_config.library_path
    assert stub.run_calls == ["stdio"]


def test_serve_is_default_action(env_config: BookRagConfig, monkeypatch: pytest.MonkeyPatch):
    """不带子命令时默认执行 serve。"""
    stub = _StubMcp()
    monkeypatch.setattr(cli, "create_mcp", lambda config: stub)

    assert cli.main([]) == 0
    assert stub.run_calls == ["stdio"]


def test_serve_respects_http_transport(
    env_config: BookRagConfig, monkeypatch: pytest.MonkeyPatch
):
    """--transport http 时以 http 传输启动。"""
    stub = _StubMcp()
    monkeypatch.setattr(cli, "create_mcp", lambda config: stub)

    assert cli.main(["--transport", "http", "serve"]) == 0
    assert stub.run_calls == ["http"]
