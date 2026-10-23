# AGENTS.md

## 项目定位
一句话：Book-RAG 是无状态书籍知识检索 MCP 服务，把本地书籍（EPUB/TXT/Markdown）解析成
可检索的知识源，通过 MCP 暴露给任意 Agent；只提供细粒度原子工具，循环思考/生成由外层 Agent 负责。

> 只收能直接读取文字的格式，**不支持 PDF**（需 OCR / 版面还原）；引用粒度到章节 + position，不含页码。

## 编排纪律（接入 Agent 必读）
你接入了一个「书籍知识检索」服务（Book-RAG），回答关于某本书的问题时遵循以下纪律：

1. 先 list_indexed_books 确认书已索引，再 get_book_structure 拿目录树，判断问题落在哪个章节。
2. 定位到章节后，优先 search_within_section 在该章节内精检索；只有范围不明时才用全局的 search_book_knowledge。
3. 需要完整上下文时，用 get_section_content 拉取章节原文；超长章按返回的 offset 续取直到章末（内容全部来自书籍本身）。
4. 「核心框架 / 方法论 / 心智模型 / 术语定义」类问题优先 retrieve_concepts；小说剧情可用 get_book_summaries 定位，摘要必须回到原文核对后引用。
5. 根据原文是否切题判断检索质量；RRF 融合分与 rerank 分不可比较，也不是相关概率，不套用统一数值阈值。结果明显不相关时，改写查询词或换检索范围再试，最多 2 次——这是纠正式检索，不是放弃。
6. 跨章节的「概念关系 / 推导 / 对比」题，拆成多个子查询分别检索后再综合。跨书题先用 books 列表与 group_by_book 分书取材，再各自 get_book_structure 定位和精检索，最后由接入 Agent 自行综合；服务端不合成答案。
7. 生成答案必须基于检索到的原文，并标注来源（source / section / position）；检索不到就明说「书中未检索到」，绝不编造、绝不用通识补全。

## 工具清单
| 工具名 | 说明 |
|---|---|
| list_indexed_books | 列出已索引书籍及语言、块数 |
| get_book_structure | 返回书籍目录树（Markdown 缩进列表），先定位章节 |
| search_within_section | 在指定章节内精检索原文片段 |
| get_section_content | 读取章节原文快照，每次最多 15000 字，按 offset 续取 |
| search_book_knowledge | 全局按内容检索，返回原文片段 + 相关度 + 来源 |
| retrieve_concepts | 检索框架/术语/心智模型条目（需离线概念索引，未建时返回提示） |
| get_book_summaries | 只读章节梗概/知识摘要及原文指针；按 offset 续取，摘要需核对原文 |
| search_by_topic | 按离线主题名/描述取原文成员并按书分组；单层聚类不等于跨书主题对齐 |
| refresh_book_index | 增量同步书籍库索引 |

## 铁律
- 绝不编造：所有返回内容锚定原文，检索不到明说「书中未检索到」，禁止通识补全。
- 结构层不内置 LLM 循环；改写查询/拆子问题/综合由接入 Agent 自己执行。
