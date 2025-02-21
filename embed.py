import json
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------
# TẢI MÔ HÌNH VIỆT NAM "dangvantuan/vietnamese-embedding"
# ---------------------------
model = SentenceTransformer("dangvantuan/vietnamese-embedding", device="cpu", trust_remote_code=True)

# ---------------------------
# HÀM TẠO EMBEDDING
# ---------------------------
def get_embedding(text):
    """
    Tạo embedding cho văn bản đầu vào sử dụng `dangvantuan/vietnamese-embedding`.
    """
    try:
        embedding = model.encode(text, normalize_embeddings=True)  # Normalize vector
        return embedding.astype("float32")  # Trả về numpy array
    except Exception as e:
        print(f"Lỗi khi tạo embedding: {e}")
        return None

# ---------------------------
# ĐỌC DỮ LIỆU Q&A TỪ FILE chatbot.json
# ---------------------------
data_file = "chatbot.json"
with open(data_file, "r", encoding="utf-8") as f:
    qa_data = json.load(f)

# ---------------------------
# TẠO EMBEDDING VÀ LƯU VÀO FILE
# ---------------------------
embedding_data = {}

# Lấy toàn bộ câu hỏi để batch encode
all_questions = [question for questions in qa_data.values() for question in questions]

# Batch encode để tăng tốc
batch_size = 16  # Điều chỉnh batch size tùy vào RAM
embeddings = model.encode(all_questions, normalize_embeddings=True, batch_size=batch_size)

# Lưu vào dictionary
for question, embedding in zip(all_questions, embeddings):
    embedding_data[question] = embedding.tolist()  # Chuyển numpy array thành list

# Lưu kết quả vào file JSON
embedding_file = "embedding_data_vietnamese.json"
with open(embedding_file, "w", encoding="utf-8") as f:
    json.dump(embedding_data, f, ensure_ascii=False, indent=4)

print(f"✅ Đã tạo embedding cho {len(embedding_data)} câu hỏi và lưu vào {embedding_file}")
