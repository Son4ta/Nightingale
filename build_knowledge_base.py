# build_knowledge_base.py (Final Refactored Version)

import os
import base64
from PIL import Image
import io
from tqdm import tqdm
import sys

import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import OpenAI

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# --- 1. 配置区域  ---
load_dotenv()

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
if not SILICONFLOW_API_KEY:
    raise ValueError("SILICONFLOW_API_KEY not found in environment variables.")

# 开关：是否处理图片描述。API太慢时可设为 False
PROCESS_IMAGES = False

SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"

# 更新为新的模型和路径
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
VLM_MODEL = "Qwen/Qwen2.5-VL-32B-Instruct"
CORPUS_DIRECTORY = "./corpus"
FAISS_INDEX_PATH = "knowledge_base"

# 更新为新的分块设置
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128

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

# --- 3. 核心函数 (已更新日志功能) ---

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
            timeout=120, # 稍微增加超时时间以应对大模型
        )
        description = response.choices[0].message.content
        return f"[图片描述: {description}]"
    except Exception as e:
        tqdm.write(f"\n[Warning] Image description API call failed: {e}. Skipping this image.")
        return "[图片处理失败]"


def load_pdfs_with_images(directory: str) -> list[Document]:
    all_docs = []
    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    
    # 使用tqdm来显示总体文件处理进度
    for filename in tqdm(pdf_files, desc="Processing PDF files"):
        file_path = os.path.join(directory, filename)
        doc = fitz.open(file_path)
        
        # <--- 新增功能: 预先统计总图片数 --->
        total_images_in_doc = 0
        if PROCESS_IMAGES:
            total_images_in_doc = sum(len(page.get_images(full=True)) for page in doc)
            if total_images_in_doc > 0:
                # tqdm.write 可以在不打乱进度条的情况下打印信息
                tqdm.write(f"  📄 在 '{filename}' 中发现 {total_images_in_doc} 张图片。")
        
        processed_image_count = 0
        # <--- 新增功能结束 --->

        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            
            if PROCESS_IMAGES:
                full_page_content = text
                image_list = page.get_images(full=True)
                if image_list:
                    image_descriptions = []
                    for img_index, img in enumerate(image_list):
                        processed_image_count += 1 # <--- 新增功能: 更新计数器
                        
                        # <--- 新增功能: 打印实时图片处理进度 --->
                        progress_message = f"\r    -> 正在处理第 {processed_image_count} / {total_images_in_doc} 张图片..."
                        sys.stdout.write(progress_message)
                        sys.stdout.flush()
                        # <--- 新增功能结束 --->
                        
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        description = get_image_description(image_bytes)
                        image_descriptions.append(description)

                    if image_descriptions:
                        full_page_content += "\n\n" + "\n".join(image_descriptions)
            else:
                full_page_content = text
            
            doc_obj = Document(
                page_content=full_page_content,
                metadata={"source": filename, "page": page_num + 1}
            )
            all_docs.append(doc_obj)
        
        # <--- 新增功能: 清理当前文件的进度行 --->
        if PROCESS_IMAGES and total_images_in_doc > 0:
            sys.stdout.write("\r" + " " * (len(progress_message) + 5) + "\r") # 清除行
            sys.stdout.flush()
            tqdm.write(f"  ✔️  '{filename}' 中的图片处理完毕。")
        # <--- 新增功能结束 --->

    return all_docs

# --- 4. 主流程 (保持不变) ---
if __name__ == "__main__":
    logger = ProgressLogger()
    
    print("Starting knowledge base creation process...")
    if not PROCESS_IMAGES:
        print("💡 Image processing is disabled.")

    logger.start("Step 1: Loading PDFs and processing content")
    documents = load_pdfs_with_images(CORPUS_DIRECTORY)
    logger.done()
    print(f"   └── Total pages loaded as documents: {len(documents)}")

    logger.start("Step 2: Splitting documents into chunks")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    logger.done()
    print(f"   └── Total non-empty chunks created: {len(chunks)}")

    logger.start("Step 3: Embedding chunks and building FAISS index")
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=SILICONFLOW_API_KEY,
        openai_api_base=SILICONFLOW_BASE_URL,
    )
    db = FAISS.from_documents(chunks, embeddings)
    logger.done()

    logger.start("Step 4: Saving index to local disk")
    db.save_local(FAISS_INDEX_PATH)
    logger.done()
    
    print(f"\n🎉 Knowledge base build process complete! Index saved to '{FAISS_INDEX_PATH}'.")