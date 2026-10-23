"""S0 结构体检：扫描待入库书籍，诊断章节结构 / 清洗 / 切分 / 转 Markdown 可行性。

**只读脚本**：不建索引、不改动书籍文件、不写 books/ 目录。输出 Markdown 报告
与控制台汇总表，供入库前决策使用。

诊断项：
1. 结构可用性：section 数、单 section 最大字符数、TOC 树深度与顶层节点数；
2. 清洗影响：front/back matter 剥离条数与字符占比（小说尤其要看有没有误伤正文）；
3. 切分规模：按当前 chunk_size / chunk_overlap 会产出多少 chunk；
4. EPUB 转 Markdown 可行性：nav/ncx 目录节点数与 spine 文档数是否一一对应；
5. 红旗：整本一个 section、单章过大、套装合一、清洗剥离过度等。

用法：
    uv run python scripts/inspect_library.py "C:/Users/joeji/Desktop/Transfer/书本"
    uv run python scripts/inspect_library.py books "C:/Users/joeji/Desktop/Transfer/书本" \
        --md docs/reports/library-structure-inspection.md --json /tmp/inspect.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from book_rag.io.cleaning import clean_matter
from book_rag.io.loader import discover_books, load_book
from book_rag.rag.language import detect_language, normalize_language
from book_rag.rag.splitter import split_book
from book_rag.rag.structure import build_toc

#: 单 section 字符数红线：超过即认为「整章过大」，get_section_content 会撑爆上下文
HUGE_SECTION_CHARS = 20_000
#: 单 section 字符数黄线
LARGE_SECTION_CHARS = 6_000
#: section 数红线：少于此值认为结构不可用（整本一个 section 之类）
MIN_SECTIONS = 5
#: 套装 / 合集关键词（一本书文件里装了多本书，需要拆分）
_OMNIBUS_KEYWORDS = ("套装", "三册", "合集", "全集", "文集", "套装共", "上册", "下册", "套装全")

# 与 BookRagConfig 默认值保持一致
_DEFAULT_CHUNK_SIZE = 512
_DEFAULT_CHUNK_OVERLAP = 64


@dataclass
class BookDiagnosis:
    """单本书的体检结果。"""

    path: str
    suffix: str
    size_mb: float
    title: str = ""
    language: str = ""
    error: str = ""

    sections: int = 0
    total_chars: int = 0
    max_section_chars: int = 0
    p50_section_chars: int = 0
    p90_section_chars: int = 0
    largest_section_path: str = ""

    toc_top_nodes: int = 0
    toc_max_depth: int = 0
    toc_leaf_nodes: int = 0

    cleaned_sections: int = 0
    removed_count: int = 0
    removed_chars: int = 0
    removed_ratio: float = 0.0
    clean_aborted: bool = False
    removed_preview: list[str] = field(default_factory=list)

    chunk_count: int = 0

    epub_spine_docs: int = 0
    epub_toc_nodes: int = 0
    epub_has_nav: bool = False

    flags: list[str] = field(default_factory=list)

    @property
    def red(self) -> bool:
        """是否存在阻断性问题（不修就不能入库）。"""
        return any(f.startswith("🔴") for f in self.flags)


def _walk_toc(node: dict, depth: int, stats: dict) -> None:
    """递归统计目录树：最大深度、叶子节点数。"""
    children = node.get("children") or []
    stats["max_depth"] = max(stats["max_depth"], depth)
    if not children:
        stats["leaves"] += 1
    for child in children:
        _walk_toc(child, depth + 1, stats)


def _epub_structure(path: Path) -> tuple[int, int, bool]:
    """返回 EPUB 的 (spine 文档数, 目录节点总数, 是否有 nav/ncx 目录)。"""
    try:
        from ebooklib import epub

        reader = epub.read_epub(str(path))
        spine = [entry for entry, _ in (reader.spine or [])]
        raw_toc = list(reader.toc or [])
    except Exception:  # noqa: BLE001 - 破损 EPUB 不阻断体检，由 parse 阶段报错
        return 0, 0, False

    def count_nodes(nodes) -> int:
        total = 0
        for node in nodes:
            total += 1
            if isinstance(node, tuple):
                total += count_nodes(list(node[1]))
        return total

    return len(spine), count_nodes(raw_toc), bool(raw_toc)


def diagnose(path: Path) -> BookDiagnosis:
    """诊断单本书；解析失败时把错误记入 ``error`` 并返回。"""
    diag = BookDiagnosis(
        path=str(path),
        suffix=path.suffix.lower(),
        size_mb=round(path.stat().st_size / 1024 / 1024, 2),
    )

    if diag.suffix == ".epub":
        spine, toc_nodes, has_nav = _epub_structure(path)
        diag.epub_spine_docs = spine
        diag.epub_toc_nodes = toc_nodes
        diag.epub_has_nav = has_nav

    try:
        parsed = load_book(path)
    except Exception as exc:  # noqa: BLE001 - 单本失败不影响整批
        diag.error = f"{type(exc).__name__}: {exc}"
        diag.flags.append("🔴 解析失败")
        return diag

    diag.title = parsed.title
    sections = parsed.sections
    diag.sections = len(sections)
    lengths = [len(s.text) for s in sections]
    diag.total_chars = sum(lengths)
    if lengths:
        diag.max_section_chars = max(lengths)
        diag.p50_section_chars = int(statistics.median(lengths))
        diag.p90_section_chars = int(statistics.quantiles(lengths, n=10)[-1]) if len(
            lengths
        ) >= 10 else max(lengths)
        diag.largest_section_path = sections[lengths.index(diag.max_section_chars)].path

    # 语言：取前 3000 字符采样（与索引层 1000 字符采样同思路，样本放大避免误判）
    sample = "\n".join(s.text for s in sections[:8])[:3000]
    diag.language = normalize_language(detect_language(sample))

    toc = build_toc(sections)
    stats = {"max_depth": 0, "leaves": 0}
    _walk_toc(toc, 0, stats)
    diag.toc_top_nodes = len(toc.get("children") or [])
    diag.toc_max_depth = stats["max_depth"]
    diag.toc_leaf_nodes = stats["leaves"]

    # 清洗影响
    try:
        cleaned, report = clean_matter(parsed)
        diag.cleaned_sections = len(cleaned.sections)
        diag.removed_count = len(report.removed)
        diag.removed_chars = sum(item.chars for item in report.removed)
        diag.removed_ratio = (
            round(diag.removed_chars / diag.total_chars, 4) if diag.total_chars else 0.0
        )
        diag.clean_aborted = report.aborted
        diag.removed_preview = [
            f"{item.kind}:{item.path}({item.reason}, {item.chars}字)" for item in report.removed[:6]
        ]
    except Exception as exc:  # noqa: BLE001 - 清洗异常只记红旗
        diag.flags.append(f"🔴 清洗异常 {type(exc).__name__}: {exc}")

    # 切分规模（用清洗后的 sections，接近真实入库形态）
    try:
        docs = split_book(
            cleaned.sections,
            chunk_size=_DEFAULT_CHUNK_SIZE,
            chunk_overlap=_DEFAULT_CHUNK_OVERLAP,
            title=parsed.title,
            source=path.name,
        )
        diag.chunk_count = len(docs)
    except Exception as exc:  # noqa: BLE001
        diag.flags.append(f"🔴 切分异常 {type(exc).__name__}: {exc}")

    _apply_flags(diag)
    return diag


def _apply_flags(diag: BookDiagnosis) -> None:
    """按体检数据补充红旗 / 黄旗诊断结论。"""
    if diag.sections and diag.sections < MIN_SECTIONS:
        diag.flags.append(
            f"🔴 结构不可用：仅 {diag.sections} 个 section（整本可能是一个文档，"
            "章节内检索与整章拉取都会退化）"
        )
    if diag.toc_top_nodes == 1 and diag.toc_max_depth <= 1:
        diag.flags.append("🔴 目录单节点：TOC 只有一根，章节定位失效")
    if diag.max_section_chars > HUGE_SECTION_CHARS:
        diag.flags.append(
            f"🔴 单章过大：{diag.max_section_chars} 字（>{HUGE_SECTION_CHARS}），"
            f"get_section_content 会撑爆上下文 · {diag.largest_section_path[:40]}"
        )
    elif diag.max_section_chars > LARGE_SECTION_CHARS:
        diag.flags.append(
            f"🟡 单章偏大：{diag.max_section_chars} 字（>{LARGE_SECTION_CHARS}）"
        )
    if any(k in diag.title or k in diag.path for k in _OMNIBUS_KEYWORDS):
        diag.flags.append("🟡 疑似套装/合集：一个文件装多本书，需拆分或按合集处理")
    if diag.removed_ratio > 0.3:
        diag.flags.append(f"🟡 清洗剥离过多：{diag.removed_ratio:.0%} 字符被剥离，需人工复核")
    if diag.clean_aborted:
        diag.flags.append("🟡 清洗已 fail-open（会清空全书，已放弃剥离）")
    if diag.suffix == ".epub":
        if not diag.epub_has_nav:
            diag.flags.append("🟡 EPUB 无 nav/ncx 目录：章节名只能靠 heading 兜底 → 转 MD 需自行切章")
        elif diag.epub_spine_docs and diag.epub_toc_nodes > diag.epub_spine_docs * 1.5:
            diag.flags.append(
                f"🟡 nav 节点({diag.epub_toc_nodes})远多于 spine 文档"
                f"({diag.epub_spine_docs})：同一文档多章，转 MD 需文档内切分"
            )
    if diag.language in ("unknown", ""):
        diag.flags.append("🟡 语言检测未命中：会回退到多语言 embedding")
    if diag.chunk_count == 0:
        diag.flags.append("🔴 清洗后无可索引文本块")


def _md_cell(text: str) -> str:
    """Markdown 表格单元格转义。"""
    return str(text).replace("|", "\\|").replace("\n", " ")


def render_markdown(diags: list[BookDiagnosis], roots: list[Path]) -> str:
    """把体检结果渲染为 Markdown 报告。"""
    total_chunks = sum(d.chunk_count for d in diags)
    reds = [d for d in diags if d.red]
    lines: list[str] = [
        "# 书库结构体检报告（S0）",
        "",
        f"- 扫描目录：{'、'.join(f'`{r}`' for r in roots)}",
        f"- 书籍数：{len(diags)}　阻断性问题书数：{len(reds)}　预计总 chunk：{total_chunks}",
        (
            f"- 判定口径：section < {MIN_SECTIONS}、目录单节点、"
            f"单章 > {HUGE_SECTION_CHARS} 字 = 🔴；"
            f"套装、单章 > {LARGE_SECTION_CHARS} 字、清洗剥离 > 30%、无 nav = 🟡"
        ),
        "",
        "## 汇总表",
        "",
        "| # | 书籍 | 格式 | MB | 语言 | section | 最大章字数 | 目录层/顶层 | 清洗剥离 | chunk | 状态 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, d in enumerate(diags, start=1):
        status = "🔴" if d.red else ("🟡" if d.flags else "✅")
        if d.error:
            status = "💥"
        lines.append(
            f"| {i} | {_md_cell(Path(d.path).name[:34])} | {d.suffix.lstrip('.')} | "
            f"{d.size_mb} | {d.language} | {d.sections} | {d.max_section_chars} | "
            f"{d.toc_max_depth}/{d.toc_top_nodes} | {d.removed_count}项/{d.removed_ratio:.1%} | "
            f"{d.chunk_count} | {status} |"
        )

    lines += ["", "## 逐本详情", ""]
    for i, d in enumerate(diags, start=1):
        lines += [f"### {i}. {Path(d.path).name}", ""]
        lines.append(f"- 路径：`{d.path}`")
        if d.error:
            lines += [f"- ❌ 解析失败：`{d.error}`", ""]
            continue
        lines += [
            f"- 书名 / 语言：{d.title} / {d.language}",
            (
                f"- 结构：{d.sections} 个 section，共 {d.total_chars:,} 字；"
                f"单章 中位 {d.p50_section_chars} / P90 {d.p90_section_chars}"
                f" / 最大 {d.max_section_chars}"
            ),
            f"- 目录树：{d.toc_top_nodes} 个顶层节点，最大深度 {d.toc_max_depth}，叶子 {d.toc_leaf_nodes}",
            (
                f"- 清洗：保留 {d.cleaned_sections}/{d.sections}，"
                f"剥离 {d.removed_count} 项 / {d.removed_chars:,} 字（{d.removed_ratio:.1%}）"
            ),
            f"- 切分：{d.chunk_count} chunk（{_DEFAULT_CHUNK_SIZE}/{_DEFAULT_CHUNK_OVERLAP}）",
        ]
        if d.suffix == ".epub":
            lines.append(
                f"- EPUB：spine 文档 {d.epub_spine_docs}，nav 节点 {d.epub_toc_nodes}"
                f"（{'有' if d.epub_has_nav else '无'}目录）"
            )
        if d.removed_preview:
            lines.append("- 剥离明细：" + "；".join(_md_cell(x) for x in d.removed_preview))
        if d.flags:
            lines.append("- 诊断：" + "；".join(d.flags))
        else:
            lines.append("- 诊断：✅ 结构健康，可直接入库")
        lines.append("")

    return "\n".join(lines)


def _print_summary(diags: list[BookDiagnosis]) -> None:
    """控制台打印紧凑汇总表。"""
    header = f"{'#':>2} {'书籍':<30} {'fmt':<5} {'sec':>5} {'max章':>7} {'nav':>5} {'chunk':>6} 状态"
    print(header)
    print("-" * len(header))
    for i, d in enumerate(diags, start=1):
        name = Path(d.path).name[:28]
        status = "RED" if d.red else ("YEL" if d.flags else "ok")
        if d.error:
            status = "ERR"
        print(
            f"{i:>2} {name:<30} {d.suffix.lstrip('.'):<5} {d.sections:>5} "
            f"{d.max_section_chars:>7} {d.epub_toc_nodes:>5} {d.chunk_count:>6} {status}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="S0 书库结构体检（只读）")
    parser.add_argument("paths", nargs="+", help="待体检的目录或书籍文件")
    parser.add_argument("--md", help="Markdown 报告输出路径")
    parser.add_argument("--json", dest="json_out", help="JSON 结果输出路径")
    args = parser.parse_args()

    roots = [Path(p) for p in args.paths]
    files: list[Path] = []
    for root in roots:
        if root.is_dir():
            # 复用 loader 的发现逻辑：自动跳过 _originals / .book_rag_index 等内部目录
            files.extend(discover_books(root))
        elif root.is_file():
            files.append(root)

    files = sorted({p.resolve() for p in files})
    if not files:
        print("未找到受支持格式的书籍文件（EPUB / TXT / Markdown）")
        return 1

    diags: list[BookDiagnosis] = []
    for path in files:
        print(f"体检中：{path.name}", file=sys.stderr)
        diags.append(diagnose(path))

    _print_summary(diags)

    if args.md:
        out = Path(args.md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_markdown(diags, roots), encoding="utf-8")
        print(f"\nMarkdown 报告：{out}")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps([asdict(d) for d in diags], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"JSON 结果：{out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
