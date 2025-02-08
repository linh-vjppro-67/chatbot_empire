import json
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------
# TẢI MÔ HÌNH E5-BASE
# ---------------------------
model = SentenceTransformer("intfloat/multilingual-e5-base", device="cpu")  # Chạy trên GPU Apple Silicon (MPS)

# ---------------------------
# HÀM TẠO EMBEDDING SỬ DỤNG E5-BASE
# ---------------------------
def get_embedding(text):
    """
    Tạo embedding cho văn bản đầu vào sử dụng `intfloat/multilingual-e5-base`.
    """
    try:
        text = f"query: {text.lower()}"  # E5 yêu cầu thêm tiền tố "query: "
        embedding = model.encode(text, normalize_embeddings=True)  # Normalize vector
        return embedding.astype("float32")  # Trả về dạng numpy array
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
all_questions = [f"query: {question.lower()}" for key in qa_data for question in qa_data[key]]

# Batch encode để tăng tốc
batch_size = 16  # Điều chỉnh batch size tùy vào RAM
embeddings = model.encode(all_questions, normalize_embeddings=True, batch_size=batch_size)

# Lưu vào dictionary
index = 0
for key, questions in qa_data.items():
    for question in questions:
        embedding_data[question] = embeddings[index].tolist()  # Chuyển numpy array thành list
        index += 1

# Lưu kết quả vào file JSON
embedding_file = "embedding_data_e5.json"
with open(embedding_file, "w", encoding="utf-8") as f:
    json.dump(embedding_data, f, ensure_ascii=False, indent=4)

print(f"✅ Đã tạo embedding cho {len(embedding_data)} câu hỏi và lưu vào {embedding_file}")
