import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import PGVector
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

app = FastAPI()

# Cấu hình lấy từ Environment Variables trên Render
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# Lưu ý: DATABASE_URL phải dùng địa chỉ Pooler (Port 6543) để tránh lỗi mạng
DB_URL = os.getenv("DATABASE_URL")

# 1. Khởi tạo Embeddings của Google (Rất nhẹ cho RAM của Render)
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

# 2. Kết nối Vector Database (Supabase)
vector_db = PGVector(
    connection_string=DB_URL,
    collection_name="nong_nghiep_data",
    embedding_function=embeddings,
    use_jsonb=True
)

# 3. Cấu hình Prompt chuyên gia nông nghiệp
template = """Bạn là một chuyên gia tư vấn kỹ thuật nông nghiệp. 
Hãy trả lời câu hỏi dựa trên ngữ cảnh được cung cấp dưới đây. 
Nếu không có trong ngữ cảnh, hãy nói 'Tôi không tìm thấy thông tin này'.

Ngữ cảnh: {context}
Câu hỏi: {question}
Trả lời (Chi tiết về bệnh và thuốc):"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# 4. Khởi tạo LLM Gemini 1.5 Flash (Nhanh và nhẹ)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

class QuestionRequest(BaseModel):
    message: str

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    result = qa_chain.invoke({"query": request.message})
    return {"answer": result["result"]}

@app.get("/")
async def root():
    return {"status": "AI Nông Nghiệp đang hoạt động"}