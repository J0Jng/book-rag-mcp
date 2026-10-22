"""文档解析与加载层。

对外入口：
- ``loader.discover_books`` / ``loader.load_book`` / ``loader.compute_file_hash``；
- ``parsers.get_parser`` 按后缀路由到 EPUB / TXT / Markdown 解析器；
- ``cleaning.clean_matter`` 在向量化前剥离 front/back matter。
"""
