import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from fastapi.middleware.cors import CORSMiddleware

# 1. KHỞI TẠO FASTAPI
app = FastAPI(title="Agri-AI API Service")

# Cấu hình CORS để Mobile App hoặc Web có thể gọi tới mà không bị chặn
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. CẤU HÌNH BIẾN MÔI TRƯỜNG (Trên Cloud sẽ cấu hình trong Dashboard)
GOOGLE_API_KEY = os.getenv("AIzaSyDDdowRQI0HUqmI7LHLk5a45bFNOJiFlmU")
# URL kết nối Postgres (Ví dụ: postgresql+psycopg2://user:pass@host:port/dbname)
DB_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:111789@localhost:5433/postgres")
COLLECTION_NAME = "nong_nghiep_phat_trien"

# 3. KHỞI TẠO CÁC THÀNH PHẦN AI
embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)

# Kết nối tới Vector Store hiện có trong Postgres
vector_db = PGVector(
    connection_string=DB_URL,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME
)

retriever = vector_db.as_retriever(search_kwargs={"k": 5})

# Prompt chuyên gia linh hoạt
template = """Bạn là chuyên gia nông nghiệp. Dựa vào tài liệu, hãy trả lời câu hỏi của nông dân.
CHỈ trả lời những phần được hỏi (Dấu hiệu, Nguyên nhân hoặc Cách điều trị).
Nếu hỏi chung, trả lời đầy đủ. Nếu không có thông tin, hãy báo chưa có dữ liệu.

TÀI LIỆU: {context}
CÂU HỎI: {question}

TRẢ LỜI:"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

# 4. ĐỊNH NGHĨA ENDPOINT CHO MOBILE APP
class ChatRequest(BaseModel):
    message: str

@app.get("/")
async def root():
    return {"message": "Hệ thống AI Nông nghiệp đang hoạt động!"}

@app.post("/ask")
async def ask_ai(request: ChatRequest):
    try:
        if not request.message:
            raise HTTPException(status_code=400, detail="Tin nhắn không được để trống")
        
        answer = rag_chain.invoke(request.message)
        return {"status": "success", "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)