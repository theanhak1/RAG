import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from fastapi.middleware.cors import CORSMiddleware

# 1. KHỞI TẠO FASTAPI
app = FastAPI(title="Agri-AI API Service")

# Cấu hình CORS để bạn Lộc có thể gọi API từ Mobile App
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. LẤY BIẾN MÔI TRƯỜNG (Cấu hình trên Dashboard Render)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DB_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = "agri_knowledge_google_v2" # Lưu ý khớp với tên lúc nạp lại dữ liệu

# 3. KHỞI TẠO CÁC THÀNH PHẦN AI (Dùng toàn bộ Google API để nhẹ RAM)
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview", 
    google_api_key=GOOGLE_API_KEY
)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    google_api_key=GOOGLE_API_KEY, 
    temperature=0.1
)

# Kết nối tới Vector Store trên Supabase
vector_db = PGVector(
    connection_string=DB_URL,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
    use_jsonb=True
)

retriever = vector_db.as_retriever(search_kwargs={"k": 5})

# Prompt chuyên gia nông nghiệp
template = """Bạn là một chuyên gia nông nghiệp Việt Nam am hiểu. 
Dựa vào tài liệu cung cấp, hãy trả lời câu hỏi của nông dân một cách ngắn gọn, dễ hiểu và chính xác.
Nếu tài liệu không có thông tin, hãy nói "Xin lỗi, hiện tại tôi chưa có dữ liệu về vấn đề này".

TÀI LIỆU: {context}
CÂU HỎI: {question}

TRẢ LỜI:"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Thiết lập chuỗi xử lý RAG
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt 
    | llm 
    | StrOutputParser()
)

# 4. ĐỊNH NGHĨA ENDPOINT
class ChatRequest(BaseModel):
    message: str

@app.get("/")
async def root():
    return {"status": "running", "message": "Hệ thống AI Nông nghiệp đã sẵn sàng!"}

@app.post("/ask")
async def ask_ai(request: ChatRequest):
    try:
        if not request.message:
            raise HTTPException(status_code=400, detail="Tin nhắn trống")
        
        # Gọi chuỗi RAG để lấy câu trả lời
        answer = rag_chain.invoke(request.message)
        return {"status": "success", "answer": answer}
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))