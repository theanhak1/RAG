import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. CẤU HÌNH (Thay thông tin của bạn)
os.environ["GOOGLE_API_KEY"] = "AIzaSyAtn-hAKYMLjnLtdocGG_SPh7Dhr5vByQU" 
CONNECTION_STRING = "postgresql+psycopg2://postgres:Phamtheanh2901@db.eolyawcnjbhxmkeseotw.supabase.co:5432/postgres"
COLLECTION_NAME = "agri_knowledge_google_v2"

# 2. XỬ LÝ VĂN BẢN
# Thay "du_lieu_nong_nghiep.txt" bằng tên file thực tế của bạn
loader = TextLoader("C:/Users/thean/Downloads/Bệnh cây và cách chữa.txt", encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)
chunks = text_splitter.split_documents(documents)

# 3. KHỞI TẠO EMBEDDINGS GOOGLE
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

# 4. NẠP VÀO SUPABASE
vector_db = PGVector.from_documents(
    embedding=embeddings,
    documents=chunks,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    pre_delete_collection=True # Xóa collection cũ để nạp mới hoàn toàn
)

print(f"✅ Đã nạp thành công {len(chunks)} đoạn dữ liệu lên Supabase!")