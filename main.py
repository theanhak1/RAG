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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. LẤY BIẾN MÔI TRƯỜNG TỪ RENDER
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DB_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = "nong_nghiep_chuyen_nghiep"

# Kiểm tra nếu thiếu biến môi trường
if not GOOGLE_API_KEY or not DB_URL:
    print("❌ Lỗi: Thiếu biến môi trường GOOGLE_API_KEY hoặc DATABASE_URL")

# 3. KHỞI TẠO CÁC THÀNH PHẦN AI
# Sử dụng model embedding nhẹ để tránh lỗi RAM trên Render Free
embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.1)

vector_db = PGVector(
    connection_string=DB_URL,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
    use_jsonb=True
)

retriever = vector_db.as_retriever(search_kwargs={"k": 5})

template = """Bạn là chuyên gia nông nghiệp. Dựa vào tài liệu, hãy trả lời câu hỏi của nông dân.
CHỈ trả lời những phần được hỏi. Nếu không có thông tin, hãy báo chưa có dữ liệu.

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
        print(f"Error: {str(e)}") # Log lỗi ra console để debug trên Render
        raise HTTPException(status_code=500, detail=str(e))