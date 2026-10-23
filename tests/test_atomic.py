"""Windows 临时占用不得损坏旧 JSON；重试有界。"""

import json
from pathlib import Path

import pytest

from book_rag.io import atomic


def test_temporary_replace_conflict_preserves_old_until_success(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    path.write_text('{"old": true}', encoding="utf-8")
    real_replace = Path.replace
    calls = []
    sleeps = []
    real_sleep = atomic.time.sleep

    def sleep(seconds):
        sleeps.append(seconds)
        real_sleep(seconds)

    def occupied(source, target):
        calls.append(1)
        if len(calls) <= 2:
            assert json.loads(path.read_text()) == {"old": True}
            raise PermissionError("temporarily occupied")
        return real_replace(source, target)

    # 只替换模块依赖的 os 对象，避免改变 pathlib 的平台选择。
    from types import SimpleNamespace

    monkeypatch.setattr(atomic, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(Path, "replace", occupied)
    monkeypatch.setattr(atomic.time, "sleep", sleep)
    atomic.write_json_atomic(path, {"new": True})
    assert json.loads(path.read_text()) == {"new": True}
    # 模拟的两次失败之外，Windows 扫描器也可能真实短暂占用目标。
    assert 3 <= len(calls) <= 6 and sleeps[:2] == [0.025, 0.05]
    assert len(sleeps) == len(calls) - 1
    assert list(tmp_path.iterdir()) == [path]


def test_permanent_conflict_is_bounded_and_leaves_old_file(tmp_path, monkeypatch):
    from types import SimpleNamespace

    path = tmp_path / "manifest.json"
    path.write_text('{"old": true}', encoding="utf-8")
    calls = []

    def occupied(*args):
        calls.append(1)
        raise PermissionError("still occupied")

    monkeypatch.setattr(atomic, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(Path, "replace", occupied)
    monkeypatch.setattr(atomic.time, "sleep", lambda seconds: None)
    with pytest.raises(PermissionError, match="still occupied"):
        atomic.write_json_atomic(path, {"new": True})
    assert len(calls) == 6
    assert json.loads(path.read_text()) == {"old": True}
    assert list(tmp_path.iterdir()) == [path]


def test_directory_publication_retries_transient_conflict(tmp_path, monkeypatch):
    from types import SimpleNamespace

    source = tmp_path / "staged"
    source.mkdir()
    (source / "body.txt").write_text("original", encoding="utf-8")
    target = tmp_path / "published"
    real_replace = Path.replace
    calls = []

    def occupied(path, destination):
        calls.append(1)
        if len(calls) == 1:
            assert source.is_dir() and not target.exists()
            raise PermissionError("directory occupied")
        return real_replace(path, destination)

    monkeypatch.setattr(atomic, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(Path, "replace", occupied)
    atomic.replace_with_retry(source, target)
    assert not source.exists()
    assert (target / "body.txt").read_text() == "original"
    assert 2 <= len(calls) <= 6
