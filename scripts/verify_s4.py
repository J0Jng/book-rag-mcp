"""S4 真实摘要补建与证据采集。凭据只从进程环境读取，不保存请求头。"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from book_rag.config import BookRagConfig
from book_rag.io.atomic import write_json_atomic
from book_rag.io.parsers import BookSection
from book_rag.rag.index import get_book_index_dir, load_book_index_cached, load_section_snapshot
from book_rag.rag.llm import load_llm
from book_rag.rag.retriever import _list_indexed_metas
from book_rag.rag.summarizer import _group_sections
from book_rag.rag.summary_layer import enrich_indexes


def core_hashes(config) -> dict:
    paths = [config.index_dir / "manifest.json"]
    for meta in _list_indexed_metas(config.index_dir, config.library_path):
        folder = get_book_index_dir(config.index_dir, Path(meta.book_path))
        paths.extend(folder / name for name in (
            "index.faiss", "index.pkl", "sections.json", "toc.json", "bm25.json", "meta.json",
        ))
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--regression", action="store_true")
    parser.add_argument("--book")
    args = parser.parse_args()
    config = BookRagConfig.from_env()
    report_dir = Path("docs/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    before = core_hashes(config)
    if args.build:
        start = time.perf_counter()
        statuses = enrich_indexes(config, book=args.book)
        write_json_atomic(report_dir / "s4-build.json", {
            "statuses": statuses, "seconds": time.perf_counter() - start,
            "core_unchanged": before == core_hashes(config),
        })
        print(json.dumps(statuses, ensure_ascii=False), flush=True)
    if args.regression:
        llm = load_llm()
        assert llm is not None
        # HEAD 中尚未修改的原版 S4 之前实现，用相同实际章节作独立 LLM 对照。
        old_source = subprocess.check_output(
            ["git", "show", "HEAD:src/book_rag/rag/summarizer.py"], encoding="utf-8",
        )
        namespace = {}
        exec(compile(old_source, "baseline_summarizer", "exec"), namespace)  # noqa: S102
        comparisons = []
        for meta in _list_indexed_metas(config.index_dir, config.library_path):
            if meta.genre != "knowledge" or "诺贝尔" in meta.book_path:
                continue
            folder = get_book_index_dir(config.index_dir, Path(meta.book_path))
            sections = [BookSection(**s) for s in load_section_snapshot(folder)]
            groups = _group_sections(sections, title=Path(meta.book_path).stem)
            chosen, members = next((name, members) for name, members in groups
                                   if "章" in name or "01 " in name)
            summaries = json.loads((folder / "summaries.json").read_text(encoding="utf-8"))
            current = next(s for s in summaries if s["section"] == chosen)
            baseline = namespace["summarize_sections"](members, llm, title=Path(meta.book_path).stem)
            assert baseline, meta.book_path
            comparisons.append({
                "source": meta.book_path, "section": chosen,
                "baseline": baseline, "current": current,
                "before_counts": {key: sum(len(s.get(key, [])) for s in baseline)
                                  for key in ("frameworks", "terms")},
                "after_counts": {key: len(current[key]) for key in ("frameworks", "terms")},
            })
        write_json_atomic(report_dir / "s4-knowledge-regression.json", comparisons)
        print("knowledge regression comparisons saved", flush=True)
    details = []
    audits = []
    for meta in _list_indexed_metas(config.index_dir, config.library_path):
        if args.book and args.book not in meta.book_path:
            continue
        folder = get_book_index_dir(config.index_dir, Path(meta.book_path))
        status = json.loads((folder / "summary_status.json").read_text(encoding="utf-8"))
        assert status["complete"] and status["book_hash"] == meta.book_hash, status
        summaries = json.loads((folder / "summaries.json").read_text(encoding="utf-8"))
        snapshot = load_section_snapshot(folder)
        store = load_book_index_cached(config.index_dir, Path(meta.book_path))[0]
        docs = {i: store.docstore.search(store.index_to_docstore_id[i])
                for i in range(store.index.ntotal)}
        assert len(summaries) == status["expected"]
        for entry in summaries:
            for pointer in entry["pointers"]:
                doc = docs[pointer["position"]]
                assert doc.metadata["source"] == pointer["source"] == meta.book_path
                assert doc.metadata["section"] == pointer["section"]
            if meta.genre == "fiction":
                text = "\n".join(s["text"] for s in snapshot
                                 if any(s["path"] == p["section"] for p in entry["pointers"]))
                assert all(q in text for q in entry["evidence"])
                assert all(name in text for name in entry["characters"])
        if meta.genre == "fiction":
            assert len(summaries) == 31
            assert not (folder / "concepts.faiss").exists()
            for i in (0, 14, 30):
                entry = summaries[i]
                audits.append({"summary": entry, "original": "\n".join(
                    s["text"] for s in snapshot if s["path"] == entry["section"]
                )})
        else:
            assert status["concepts_complete"]
        details.append(status)
    write_json_atomic(report_dir / "s4-integrity.json", {
        "books": details, "core_unchanged": before == core_hashes(config),
    })
    if audits:
        write_json_atomic(report_dir / "s4-fiction-audit-inputs.json", audits)
    print("S4 integrity passed", flush=True)


if __name__ == "__main__":
    main()
