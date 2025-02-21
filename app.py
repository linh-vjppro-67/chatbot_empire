import json
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer

# ---------------------------
# TẢI MÔ HÌNH E5-BASE
# ---------------------------
model = SentenceTransformer("intfloat/multilingual-e5-base", device="cpu")  # Chạy trên GPU Apple Silicon (MPS)

# ---------------------------
# HÀM TẠO EMBEDDING BATCH SỬ DỤNG E5-BASE
# ---------------------------
def get_embedding_batch(texts):
    texts = [f"query: {t.lower()}" for t in texts]  # Định dạng theo yêu cầu của E5
    embeddings = model.encode(texts, normalize_embeddings=True, batch_size=16)  # Batch xử lý nhanh hơn
    return np.array(embeddings, dtype="float32")  # Chuyển về numpy array

# ---------------------------
# ĐỌC DỮ LIỆU Q&A TỪ FILE chatbot.json
# ---------------------------
data_file = "chatbot.json"
with open(data_file, "r", encoding="utf-8") as f:
    qa_data = json.load(f)

# ---------------------------
# ĐỌC EMBEDDING TỪ FILE HOẶC TẠO MỚI
# ---------------------------
embedding_file = "embedding_data_e5.json"

try:
    with open(embedding_file, "r", encoding="utf-8") as f:
        embedding_data = json.load(f)
    print("✅ Loaded precomputed embeddings.")
except FileNotFoundError:
    print("🚀 Embedding file not found. Creating new embeddings...")
    questions_list = [q for qs in qa_data.values() for q in qs]  # Lấy tất cả câu hỏi
    embeddings_array = get_embedding_batch(questions_list)  # Encode toàn bộ danh sách câu hỏi
    embedding_data = {q: embeddings_array[i].tolist() for i, q in enumerate(questions_list)}

    # Lưu vào file JSON
    with open(embedding_file, "w", encoding="utf-8") as f:
        json.dump(embedding_data, f, ensure_ascii=False, indent=4)

# ---------------------------
# CHUYỂN EMBEDDING TỪ JSON SANG NUMPY ARRAY
# ---------------------------
questions_list = list(embedding_data.keys())
embeddings_list = [embedding_data[question] for question in questions_list]
embeddings_array = np.array(embeddings_list, dtype="float32")

# ---------------------------
# KIỂM TRA KÍCH THƯỚC EMBEDDING (E5-BASE LÀ 768)
# ---------------------------
if embeddings_array.shape[1] != 768:
    print(f"❌ Lỗi: Mô hình e5-base yêu cầu embedding 768 chiều, nhưng dữ liệu có {embeddings_array.shape[1]} chiều!")
    exit()

# ---------------------------
# CHUẨN HÓA VECTORS (L2 normalization)
# ---------------------------
faiss.normalize_L2(embeddings_array)

# ---------------------------
# XÂY DỰNG FAISS INDEX
# ---------------------------
d = 768  # E5-base có 768 chiều, không phải 1024 như bge-m3
index = faiss.IndexFlatIP(d)  # Dùng inner product (IP) vì đã normalize
index.add(embeddings_array)  # Thêm embedding vào FAISS

# Tạo mapping từ chỉ số FAISS sang câu hỏi
index_to_question = {i: questions_list[i] for i in range(len(questions_list))}

# ---------------------------
# HÀM XỬ LÝ TRUY VẤN Q&A SỬ DỤNG FAISS
# ---------------------------
def answer_query_faiss(user_query, similarity_threshold=0.2):
    query_emb = get_embedding_batch([user_query])  # Encode batch (chỉ 1 câu)
    
    k = 3  # Lấy 1 kết quả tốt nhất
    distances, indices = index.search(query_emb, k)
    
    best_score = distances[0][0]
    best_index = indices[0][0]

    for i in range(2):  # 2 queries: có dấu và không dấu
        for j in range(k):
            if distances[i][j] > best_score:  # Chọn điểm cao nhất
                best_score = distances[i][j]
                best_index = indices[i][j]

    if best_score < similarity_threshold:
        return "Chúng tôi chưa hiểu câu hỏi của bạn.", None, None

    best_question = index_to_question[best_index]
    
    for key, value in qa_data.items():
        if best_question in value:
            return key, best_question, best_score  # **Trả về câu trả lời, câu hỏi tìm thấy và độ tương đồng**
    
    return "Câu hỏi không khớp với dữ liệu, vui lòng thử lại!", None, None

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.title("Chatbot tra cứu thông tin các kỳ thi của Empire")

user_query = st.text_input("Bạn hỏi:")

if user_query:
    answer, matched_question, similarity = answer_query_faiss(user_query)

    if matched_question:

        if isinstance(answer, str) and answer.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp")):
            st.image(answer, caption="Kết quả tìm thấy", use_container_width=True)
        else:
            st.markdown(f"**Trả lời:** \n\n{answer}")
    else:
        st.markdown(f"**Trả lời:** \n\n{answer}")

