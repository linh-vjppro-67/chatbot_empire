import json
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
import unicodedata
import re

# ---------------------------
# TẢI MÔ HÌNH VIETNAMESE-EMBEDDING
# ---------------------------
model = SentenceTransformer("dangvantuan/vietnamese-embedding", device="cpu")

# ---------------------------
# HÀM LOẠI BỎ DẤU TIẾNG VIỆT
# ---------------------------
def remove_accents(text):
    text = unicodedata.normalize('NFD', text)
    text = re.sub(r'\p{Mn}', '', text)
    return text

# ---------------------------
# HÀM TẠO EMBEDDING BATCH
# ---------------------------
def get_embedding_batch(texts):
    texts = [remove_accents(t.lower()) for t in texts]  # Loại bỏ dấu trước khi embedding
    embeddings = model.encode(texts, normalize_embeddings=True, batch_size=16)
    return np.array(embeddings, dtype="float32")

# ---------------------------
# ĐỌC DỮ LIỆU Q&A TỪ FILE chatbot.json
# ---------------------------
data_file = "chatbot.json"
with open(data_file, "r", encoding="utf-8") as f:
    qa_data = json.load(f)

# ---------------------------
# ĐỌC EMBEDDING TỪ FILE HOẶC TẠO MỚI
# ---------------------------
embedding_file = "embedding_data_vn.json"

try:
    with open(embedding_file, "r", encoding="utf-8") as f:
        embedding_data = json.load(f)
    print("✅ Loaded precomputed embeddings.")
except FileNotFoundError:
    print("🚀 Embedding file not found. Creating new embeddings...")
    questions_list = [q for qs in qa_data.values() for q in qs]
    embeddings_array = get_embedding_batch(questions_list)
    embedding_data = {q: embeddings_array[i].tolist() for i, q in enumerate(questions_list)}
    
    with open(embedding_file, "w", encoding="utf-8") as f:
        json.dump(embedding_data, f, ensure_ascii=False, indent=4)

# ---------------------------
# CHUYỂN EMBEDDING TỪ JSON SANG NUMPY ARRAY
# ---------------------------
questions_list = list(embedding_data.keys())
embeddings_list = [embedding_data[question] for question in questions_list]
embeddings_array = np.array(embeddings_list, dtype="float32")

# ---------------------------
# CHUẨN HÓA VECTORS (L2 normalization)
# ---------------------------
faiss.normalize_L2(embeddings_array)

# ---------------------------
# XÂY DỰNG FAISS INDEX
# ---------------------------
d = embeddings_array.shape[1]
index = faiss.IndexFlatIP(d)
index.add(embeddings_array)

index_to_question = {i: questions_list[i] for i in range(len(questions_list))}

# ---------------------------
# HÀM XỬ LÝ TRUY VẤN Q&A SỬ DỤNG FAISS
# ---------------------------
def answer_query_faiss(user_query, similarity_threshold=0.2):
    query_emb = get_embedding_batch([user_query])
    k = 1
    distances, indices = index.search(query_emb, k)
    
    best_score = distances[0][0]
    best_index = indices[0][0]

    if best_score < similarity_threshold:
        return "Chúng tôi chưa hiểu câu hỏi của bạn.", None, None
    
    best_question = index_to_question[best_index]
    
    for key, value in qa_data.items():
        if best_question in value:
            return key, best_question, best_score
    
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
