Book-RAG: A retrieval-augmented generation system for querying full books via natural language.

- Ingest complete book texts into a searchable vector database with chapter-level indexing.
- Retrieve context-aware passages using hybrid keyword + semantic search to answer specific questions.
- Return original passages with source, section, and chunk position; answer synthesis belongs to the external agent. Page references and PDF/OCR are out of scope.
- Support many-to-many workspaces within one library. Multi-book grouping and offline topic clustering are planned for S5/S6 and are not implemented yet.
- Expose atomic MCP tools, a CLI, and a local library/search web interface; the service does not generate chat answers.
