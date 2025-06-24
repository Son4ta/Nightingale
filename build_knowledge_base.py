# build_knowledge_base.py (Optimized for Corpus Export)

import os
import base64
from PIL import Image
import io
import sys
import json

import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# --- 1. 配置区域 ---
load_dotenv()

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
if not SILICONFLOW_API_KEY:
    raise ValueError("SILICONFLOW_API_KEY not found in environment variables.")

# 开关：是否处理图片描述。API太慢时可设为 False
PROCESS_IMAGES = False

SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"

# 模型和路径配置
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
VLM_MODEL = "Qwen/Qwen2.5-VL-32B-Instruct"
CORPUS_DIRECTORY = "./corpus"
FAISS_INDEX_PATH = "knowledge_base"
# !!! 新增：定义导出语料库的文件路径 !!!
CORPUS_EXPORT_PATH = "exported_corpus.json"

# 文本分块设置
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 256

vlm_client = OpenAI(api_key=SILICONFLOW_API_KEY, base_url=SILICONFLOW_BASE_URL)

# --- 2. 进度日志模块 ---
class ProgressLogger:
    """一个简单的类，用于在同一行动态显示任务进度。"""
    def start(self, message: str):
        sys.stdout.write(f"⚙️  {message}...")
        sys.stdout.flush()

    def done(self):
        sys.stdout.write("\r✅  Done.                                  \n")
        sys.stdout.flush()

# --- 3. 核心函数 (已更新) ---

def get_image_description(image_bytes: bytes) -> str:
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    try:
        response = vlm_client.chat.completions.create(
            model=VLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "你是一个专业的看图说话助手，请用中文详细描述这幅图片的内容，重点描述图中的关键信息、图表和流程。"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=512,
            timeout=120,
        )
        description = response.choices[0].message.content
        return f"[图片描述: {description}]"
    except Exception as e:
        tqdm.write(f"\n[Warning] Image description API call failed: {e}. Skipping this image.")
        return "[图片处理失败]"


def load_pdfs_with_images(directory: str) -> list[Document]:
    all_docs = []
    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    
    for filename in tqdm(pdf_files, desc="Processing PDF files"):
        file_path = os.path.join(directory, filename)
        doc = fitz.open(file_path)
        
        total_images_in_doc = 0
        if PROCESS_IMAGES:
            total_images_in_doc = sum(len(page.get_images(full=True)) for page in doc)
            if total_images_in_doc > 0:
                tqdm.write(f"  📄 在 '{filename}' 中发现 {total_images_in_doc} 张图片。")
        
        processed_image_count = 0
        
        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            full_page_content = text
            
            if PROCESS_IMAGES and page.get_images(full=True):
                image_descriptions = []
                for img_index, img in enumerate(page.get_images(full=True)):
                    processed_image_count += 1
                    progress_message = f"\r    -> 正在处理第 {processed_image_count} / {total_images_in_doc} 张图片..."
                    sys.stdout.write(progress_message)
                    sys.stdout.flush()
                    
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    description = get_image_description(image_bytes)
                    image_descriptions.append(description)

                if image_descriptions:
                    full_page_content += "\n\n" + "\n".join(image_descriptions)

            doc_obj = Document(
                page_content=full_page_content,
                metadata={"source": filename, "page": page_num + 1}
            )
            all_docs.append(doc_obj)
        
        if PROCESS_IMAGES and total_images_in_doc > 0:
            sys.stdout.write("\r" + " " * (len(progress_message) + 5) + "\r")
            sys.stdout.flush()
            tqdm.write(f"  ✔️  '{filename}' 中的图片处理完毕。")

    return all_docs

# !!! 新增功能：将提取的完整语料保存到文件 !!!
def save_corpus_to_jsonl(documents: list[Document], file_path: str):
    """将文档列表以 JSONL 格式保存到文件。"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for doc in tqdm(documents, desc=f"Exporting corpus to {file_path}"):
            # 构建一个包含元数据和内容的字典
            record = {
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", 0),
                "content": doc.page_content
            }
            # 将字典序列化为JSON字符串并写入文件，后跟换行符
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

# --- 4. 主流程 (已更新) ---
if __name__ == "__main__":
    logger = ProgressLogger()
    
    print("Starting knowledge base creation process...")
    if not PROCESS_IMAGES:
        print("💡 Image processing is disabled.")

    logger.start("Step 1: Loading PDFs and processing content")
    documents = load_pdfs_with_images(CORPUS_DIRECTORY)
    logger.done()
    print(f"   └── Total pages (documents) loaded: {len(documents)}")

    # --- 新增步骤：在分块前导出完整语料 ---
    logger.start(f"Step 2: Exporting full corpus to '{CORPUS_EXPORT_PATH}'")
    save_corpus_to_jsonl(documents, CORPUS_EXPORT_PATH)
    logger.done()
    # --- 导出步骤结束 ---

    logger.start("Step 3: Splitting documents into chunks for RAG")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    logger.done()
    print(f"   └── Total non-empty chunks created: {len(chunks)}")

    logger.start("Step 4: Embedding chunks and building FAISS index")
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=SILICONFLOW_API_KEY,
        openai_api_base=SILICONFLOW_BASE_URL,
    )
    db = FAISS.from_documents(chunks, embeddings)
    logger.done()

    logger.start("Step 5: Saving index to local disk")
    db.save_local(FAISS_INDEX_PATH)
    logger.done()
    
    print(f"\n🎉 Knowledge base build process complete!")
    print(f"   - RAG index saved to '{FAISS_INDEX_PATH}'")
    print(f"   - Full corpus exported to '{CORPUS_EXPORT_PATH}'")