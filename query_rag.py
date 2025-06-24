# query_rag.py (Optimized with Step-Back Query Rewriting)

import os
from dotenv import load_dotenv

from langchain.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from openai import OpenAI

# --- 1. 配置区域 ---
load_dotenv()

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
if not SILICONFLOW_API_KEY:
    raise ValueError("SILICONFLOW_API_KEY not found in environment variables.")

SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
LLM_MODEL = "Qwen/Qwen3-30B-A3B"
REWRITE_MODEL = "Qwen/Qwen3-30B-A3B" 

FAISS_INDEX_PATH = "knowledge_base"
RETRIEVAL_TOP_K = 5 # 为每个检索器获取 Top K 个结果

# --- 2. RAG 核心逻辑 (已按要求优化) ---

# !!! 优化点: 新增一个辅助函数，用于调用LLM进行查询改写 !!!
def rewrite_query_with_llm(question: str) -> str:
    """
    使用大模型进行“退回式”查询改写。
    将具体问题抽象成一个更通用、更适合语义检索的问题。
    """
    client = OpenAI(api_key=SILICONFLOW_API_KEY, base_url=SILICONFLOW_BASE_URL)
    
    system_prompt = "你是一个查询优化专家。你的任务是根据用户的具体问题，生成一个更通用、更宏观的“退回式”(Step-back)问题。这个退回式问题应该捕捉原始问题的核心概念和意图，但更适合用于在大规模知识库中进行高层次的语义检索。请只返回这个退回式问题，不要任何解释或多余的文字，保持英文缩写不需要翻译"
    
    user_prompt = f"原始问题: {question}\n\n退回式问题:"

    try:
        response = client.chat.completions.create(
            model=REWRITE_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=100,
            timeout=60,
        )
        rewritten_query = response.choices[0].message.content.strip()
        return rewritten_query if rewritten_query else question
    except Exception as e:
        print(f"\n[Warning] Query rewrite failed: {e}. Falling back to original query.")
        return question

def format_docs(docs):
    """格式化检索到的文档"""
    return "\n---\n".join(
        f"来源: {doc.metadata['source']}, 页码: {doc.metadata['page']}\n内容: {doc.page_content}"
        for doc in docs
    )

print("Initializing models and retrievers...")

# 初始化嵌入模型
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

# 初始化关键词检索器 (BM25)
doc_list = list(db.docstore._dict.values())
print("Initializing BM25 keyword retriever...")
bm25_retriever = BM25Retriever.from_documents(doc_list)
bm25_retriever.k = RETRIEVAL_TOP_K

# Prompt 模板 (保持不变)
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

# 初始化 LLM (保持不变)
llm = ChatOpenAI(
    model_name=LLM_MODEL,
    openai_api_key=SILICONFLOW_API_KEY,
    openai_api_base=SILICONFLOW_BASE_URL,
    temperature=0.1,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

# 定义最终的 RAG 链
rag_chain = (
    prompt
    | llm
    | StrOutputParser()
)

# --- 3. 交互式查询 (已优化) ---
if __name__ == "__main__":
    print("\n--- RAG Exam Assistant Ready (Optimized) ---")
    print(f"   LLM Model: {LLM_MODEL}")
    print(f"   Embedding Model: {EMBEDDING_MODEL}")
    print("---------------------------------------------")
    print("Enter your question, or type 'exit' to quit.")
    
    while True:
        original_question = input("\n[考试问题] > ")
        if original_question.lower() == 'exit':
            break
        if not original_question:
            continue
            
        # !!! 优化点: 步骤 1 - 进行查询改写 !!!
        print("\n🔄 [1. 正在进行退回式查询改写...]")
        rewritten_question = rewrite_query_with_llm(original_question)
        print(f"   └─ 语义检索查询: \"{rewritten_question}\"")

        # !!! 优化点: 步骤 2 - 分别进行语义和关键词检索 !!!
        print("\n🔍 [2. 正在进行混合检索...]")
        # 使用“退回问题”进行语义搜索
        semantic_docs = faiss_retriever.invoke(rewritten_question)
        print(f"   ├─ 语义检索完成，找到 {len(semantic_docs)} 个结果。")
        # 使用“原始问题”进行关键词搜索
        keyword_docs = bm25_retriever.invoke(original_question)
        print(f"   └─ 关键词检索完成，找到 {len(keyword_docs)} 个结果。")

        # !!! 优化点: 步骤 3 - 合并并去重检索结果 !!!
        all_docs = semantic_docs + keyword_docs
        unique_docs_dict = {}
        for doc in all_docs:
            # 使用页面内容作为唯一标识符
            unique_docs_dict[doc.page_content] = doc
        
        retrieved_docs = list(unique_docs_dict.values())
        print(f"\n📚 [3. 合并去重后，共找到 {len(retrieved_docs)} 个相关上下文]")
        if retrieved_docs:
            for i, doc in enumerate(retrieved_docs):
                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page', 'N/A')
                print(f"  [片段 {i+1}] 来源: {source}, 页码: {page}")
                print(f"  内容预览: {doc.page_content[:50].replace(chr(10), ' ')}...")
        
        # 步骤 4: 基于上下文生成答案
        print("\n🧠 [4. 正在生成最终答案...]\n")
        
        context_str = format_docs(retrieved_docs)
        
        # 调用 RAG 链，流式回调会自动打印答案
        rag_chain.invoke({
            "context": context_str,
            "question": original_question # !!! 关键：用原始问题向LLM提问 !!!
        })
        print() # 在流式输出结束后换行