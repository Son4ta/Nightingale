# Nightingale: RAG for ADAI Finals of UCAS

Nightingale 是一个检索增强生成 (Retrieval-Augmented Generation, RAG) 系统，旨在将学术文档（PDF）语料库转化为一个可交互的知识库。是为国科大（UCAS）的《高级人工智能》（ADAI）课程期末考试而设计的。该系统允许用户使用自然语言提问，并获得直接从所提供的课程材料中提取的、有来源依据的精确回答。

本项目采用混合检索方法，结合了基于向量的语义搜索和传统的关键词搜索，以确保检索结果的全面性和相关性。它还包含可选的多模态处理能力，能够描述 PDF 中的图片，进一步丰富了生成答案的上下文信息。

## ✨ 主要特性

* **混合检索**: 采用一个 `EnsembleRetriever`，结合了 FAISS 的语义搜索（理解含义）和 BM25 的关键词搜索（匹配精确术语），确保了强大的上下文检索能力。
* **多模态处理**: 可选功能，能够使用视觉语言模型 (VLM) 分析 PDF 中的图片并生成文本描述，这些描述随后会被整合进知识库中。
* **答案来源引用**: LLM 被严格指示，其回答必须完全基于提供的上下文，并为使用的每一条信息引用其来源文件名和页码。
* **流式输出**: 语言模型的回答以流的形式实时返回，提供了互动和响应迅速的用户体验。
* **模块化架构**: 代码被清晰地分为两个主要脚本：一个用于构建知识库 (`build_knowledge_base.py`)，另一个用于查询 (`query_rag.py`)。
* **高度可配置**: 模型名称、分块大小和其他关键参数都在每个脚本的配置区域集中定义，便于调整。

## ⚙️ 系统工作流

系统主要分两个阶段运行：

### 阶段一：知识库构建 (`build_knowledge_base.py`)

1.  **加载 PDF**: 脚本扫描 `./corpus` 目录下的所有 PDF 文件。
2.  **内容提取**: 对每个 PDF，提取每一页的文本内容。
3.  **图片处理 (可选)**: 如果 `PROCESS_IMAGES` 设置为 `True`，脚本会提取页面中的图片，将其发送给 VLM (视觉语言模型) 获取文本描述，并将描述添加到页面内容中。
4.  **文本分块**: 使用 `RecursiveCharacterTextSplitter` 将处理后的文档（包含文本和图片描述）分割成更小的块 (chunks)。
5.  **生成向量**: 使用 SiliconFlow 的 Embedding 模型将每个文本块转换为一个数字向量 (embedding)。
6.  **FAISS 索引**: 将生成的向量存储在一个 FAISS 索引中，以实现超快的语义相似度搜索。最终的索引保存在本地的 `knowledge_base/` 目录中。

### 阶段二：查询与问答 (`query_rag.py`)

1.  **加载索引**: 脚本从 `knowledge_base/` 目录加载预先构建好的 FAISS 索引。
2.  **初始化检索器**: 初始化一个混合检索器 (`EnsembleRetriever`)，它结合了已加载的 FAISS 检索器和 BM25 关键词检索器。
3.  **用户输入**: 用户通过命令行界面输入一个问题。
4.  **上下文检索**: 用户的问题被用来查询混合检索器，从知识库中检索最相关的文本块（文档）。
5.  **构建 Prompt**: 将检索到的文档格式化后，与用户的原始问题一起注入到一个 Prompt 模板中。该模板指示 LLM 扮演一个严格的课程助教角色。
6.  **生成答案**: 完整的 Prompt 被发送给一个大语言模型。模型仅根据提供的上下文生成答案，并将带有来源引用的回答以流式输出返回给用户。

## 🛠️ 技术栈

* **核心框架**: LangChain
* **语言模型**:
    * **大语言模型 (LLM)**: `Qwen/Qwen3-30B-A3B`
    * **Embedding 模型**: `Qwen/Qwen3-Embedding-8B`
    * **视觉模型 (VLM)**: `Qwen/Qwen2.5-VL-32B-Instruct`
    * **服务接入点**: SiliconFlow API
* **向量数据库**: FAISS (`faiss-cpu`)
* **关键词检索**: `rank_bm25`
* **PDF 处理**: PyMuPDF (`fitz`)
* **开发环境**: Python, `python-dotenv`

## 🚀 快速开始

### 先决条件

* Python 3.10 或更高版本
* 一个 SiliconFlow API 密钥

### 安装与配置指南

1.  **克隆仓库:**
    ```bash
    git clone https://your-repo-url/Nightingale.git
    cd Nightingale
    ```

2.  **创建虚拟环境 (推荐):**
    ```bash
    conda create -n rag python=3.11
    ```

3.  **安装依赖:**
    `requirements.txt` 文件包含了所有必需的 Python 包。
    ```bash
    pip install -r requirements.txt
    ```

4.  **配置你的凭证:**
    在项目根目录下创建一个名为 `.env` 的文件。`.gitignore` 文件已配置为忽略此文件，以保护你的密钥安全。将你的 API 密钥添加到 `.env` 文件中：
    ```
    SILICONFLOW_API_KEY="你的_siliconflow_api_key"
    ```

### 使用指南

1.  **添加你的文档**: 将所有你希望包含在知识库中的 PDF 文件放入 `corpus/` 目录下。默认情况下，此目录被 Git 忽略。

2.  **构建知识库**: 运行 `build_knowledge_base.py` 脚本来处理你的 PDF 文件并创建 FAISS 索引。
    ```bash
    python build_knowledge_base.py
    ```
    * 如果需要包含图片分析功能（可能较慢且有成本），请在脚本中将 `PROCESS_IMAGES` 变量改为 `True`。
    * 你将看到处理 PDF 和图片（如果启用）的进度条。完成后，索引将保存在 `knowledge_base/` 文件夹中。

3.  **查询你的知识库**: 运行 `query_rag.py` 脚本以启动交互式问答助手。
    ```bash
    python query_rag.py
    ```
    * 脚本将加载模型和索引。准备就绪后，你就可以开始提问了。
    * 输入 `exit` 退出程序。

    **交互示例:**
    ```
    --- RAG Exam Assistant Ready ---
       LLM Model: Qwen/Qwen3-30B-A3B
       Embedding Model: Qwen/Qwen3-Embedding-8B
    ------------------------------------
    Enter your question, or type 'exit' to quit.

    [考试问题] > 混合检索的关键原理是什么？

    🔍 [1. 正在检索相关上下文...]

    📚 [2. 已找到以下上下文信息:]
    ========================================
      [片段 1] 来源: lecture_slides.pdf, 页码: 12
      内容: 混合检索结合了语义搜索的优势...
    ----------------------------------------
      [片段 2] 来源: course_notes.pdf, 页码: 5
      内容: BM25 对于关键词匹配很有效，而密集向量则能捕捉上下文含义...
    ----------------------------------------

    🧠 [3. 正在生成最终答案...]

    混合检索的工作原理是结合两种主要方法：关键词搜索和语义搜索。关键词方法，如 BM25，非常适合查找包含查询中精确术语的文档 (来源: course_notes.pdf, 页码: 5)。语义搜索使用向量嵌入来查找在上下文含义上相似的文档，即使它们不使用完全相同的词语 (来源: lecture_slides.pdf, 页码: 12)。通过对两种方法的结果进行加权和组合，系统可以实现更准确、更全面的检索。
    ```

## 🔧 参数配置

你可以通过修改脚本中配置区域的变量来调整系统的行为：

### `build_knowledge_base.py`

* `PROCESS_IMAGES`: `True` 或 `False`。开启或关闭图片描述处理。
* `VLM_MODEL`: 用于图片描述的模型。
* `CHUNK_SIZE`: 每个文本块的最大字符数。
* `CHUNK_OVERLAP`: 相邻文本块之间重叠的字符数。

### `query_rag.py`

* `LLM_MODEL`: 用于生成答案的主要语言模型。
* `RETRIEVAL_TOP_K`: 从每个检索器（BM25 和 FAISS）中检索的文档数量。
* `weights`: 在 `EnsembleRetriever` 的定义中，你可以调整给予 BM25 和 FAISS 结果的权重。默认为 `[0.5, 0.5]`。