# query_rag.py (Refactored to show context)

import os
from dotenv import load_dotenv

from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# --- 1. 配置区域 (已按您的要求更新) ---
load_dotenv()

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
if not SILICONFLOW_API_KEY:
    raise ValueError("SILICONFLOW_API_KEY not found in environment variables.")

# 更新为 SiliconFlow 的 v1 地址
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"

# 更新为 Qwen3 系列模型
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
LLM_MODEL = "Qwen/Qwen3-30B-A3B"

# 更新为新的索引路径
FAISS_INDEX_PATH = "knowledge_base"
RETRIEVAL_TOP_K = 5

# --- 2. RAG 核心逻辑 (已升级为混合检索) ---

print("Initializing models and retrievers...")
embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    openai_api_key=SILICONFLOW_API_KEY,
    openai_api_base=SILICONFLOW_BASE_URL
)

# 加载FAISS向量检索器
try:
    db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    faiss_retriever = db.as_retriever(search_kwargs={"k": RETRIEVAL_TOP_K})
except Exception as e:
    print(f"Fatal: Could not load FAISS index from '{FAISS_INDEX_PATH}'. Error: {e}")
    exit()

# !!! 新增：设置关键词检索器 (BM25) !!!
# BM25 需要原始文档，我们从 FAISS 的 docstore 中获取
# 这假设 FAISS 索引是用 from_documents 构建的，我们的脚本正是如此
doc_list = list(db.docstore._dict.values())

print("Initializing BM25 keyword retriever...")
bm25_retriever = BM25Retriever.from_documents(doc_list)
bm25_retriever.k = RETRIEVAL_TOP_K

# 定义最终的检索器
print("Initializing Ensemble (Hybrid) Retriever...")
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.5, 0.5]  # 权重可以调整，这里是关键词和语义各占一半
)


# Prompt 模板
template = """
你是一个严谨的课程助教，你的任务是根据我提供的上下文材料来回答问题。

**指令**:
1.  **严格依据材料**: 你的回答必须完全基于以下提供的上下文材料，禁止使用任何外部知识。
2.  **精确引用**: 对你回答中的每一个观点或数据，都必须在句末用 `(来源: <文件名>, 页码: <页码>)` 的格式注明出处。如果一个观点综合了多个来源，请全部列出。
3.  **全面而简洁**: 综合所有相关信息，给出全面而精炼的回答。
4.  **未知则答无**: 如果上下文材料不足以回答问题，必须明确回答：“根据提供的材料，无法回答此问题。”

**上下文材料**:
---
{context}
---

**问题**: {question}

**你的回答**:
"""
prompt = PromptTemplate.from_template(template)

# 初始化 LLM，使用流式输出
llm = ChatOpenAI(
    model_name=LLM_MODEL,
    openai_api_key=SILICONFLOW_API_KEY,
    openai_api_base=SILICONFLOW_BASE_URL,
    temperature=0.1,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

def format_docs(docs):
    """格式化检索到的文档，以便清晰地展示给 LLM"""
    return "\n---\n".join(
        f"来源: {doc.metadata['source']}, 页码: {doc.metadata['page']}\n内容: {doc.page_content}"
        for doc in docs
    )

# 定义 RAG 链
rag_chain = (
    {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- 3. 交互式查询 (已更新，会显示上下文) ---
if __name__ == "__main__":
    print("\n--- RAG Exam Assistant Ready ---")
    print(f"   LLM Model: {LLM_MODEL}")
    print(f"   Embedding Model: {EMBEDDING_MODEL}")
    print("------------------------------------")
    print("Enter your question, or type 'exit' to quit.")
    
    while True:
        question = input("\n[考试问题] > ")
        if question.lower() == 'exit':
            break
        if not question:
            continue
            
        # 步骤 1: 检索上下文
        print("\n🔍 [1. 正在检索相关上下文...]")
        retrieved_docs = ensemble_retriever.invoke(question)
        
        # 步骤 2: 打印检索到的上下文内容
        print("\n📚 [2. 已找到以下上下文信息:]")
        print("="*40)
        if not retrieved_docs:
            print("   (未找到相关上下文)")
        else:
            for i, doc in enumerate(retrieved_docs):
                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page', 'N/A')
                print(f"  [片段 {i+1}] 来源: {source}, 页码: {page}")
                print(f"  内容: {doc.page_content[:250]}...") # 打印内容的前250个字符作为预览
                print("-"*40)
        
        # 步骤 3: 基于上下文生成答案
        print("\n🧠 [3. 正在生成最终答案...]\n")
        # 调用 RAG 链，流式回调会自动打印答案
        rag_chain.invoke(question)
        print() # 在流式输出结束后换行