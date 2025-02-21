import json
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from unidecode import unidecode  # Thư viện bỏ dấu tiếng Việt

# ---------------------------
# 1️⃣ TẢI MÔ HÌNH E5-BASE
# ---------------------------
model = SentenceTransformer("intfloat/multilingual-e5-base", device="cpu")  # Chạy trên CPU

# ---------------------------
# 2️⃣ HÀM TẠO EMBEDDING BATCH
# ---------------------------
def get_embedding_batch(texts):
    texts = [f"query: {t.lower()}" for t in texts]  # Định dạng theo yêu cầu của E5
    embeddings = model.encode(texts, normalize_embeddings=True, batch_size=16)  # Batch xử lý nhanh hơn
    return np.array(embeddings, dtype="float32")  # Chuyển về numpy array

# ---------------------------
# 3️⃣ ĐỌC DỮ LIỆU Q&A TỪ FILE chatbot.json
# ---------------------------
data_file = "chatbot.json"
with open(data_file, "r", encoding="utf-8") as f:
    qa_data = json.load(f)

# ---------------------------
# 4️⃣ XỬ LÝ EMBEDDING: LƯU CẢ CÂU HỎI CÓ DẤU VÀ KHÔNG DẤU
# ---------------------------
embedding_file = "embedding_data_e5.json"

try:
    with open(embedding_file, "r", encoding="utf-8") as f:
        embedding_data = json.load(f)
    print("✅ Loaded precomputed embeddings.")
except FileNotFoundError:
    print("🚀 Embedding file not found. Creating new embeddings...")
    
    # Lấy danh sách câu hỏi có dấu và tạo bản không dấu
    questions_list = [q for qs in qa_data.values() for q in qs]  
    questions_list_no_accent = [unidecode(q) for q in questions_list]  

    # Gộp cả hai danh sách vào FAISS
    all_questions = questions_list + questions_list_no_accent  
    embeddings_array = get_embedding_batch(all_questions)  

    # Lưu vào dictionary
    embedding_data = {q: embeddings_array[i].tolist() for i, q in enumerate(all_questions)}

    # Lưu file JSON
    with open(embedding_file, "w", encoding="utf-8") as f:
        json.dump(embedding_data, f, ensure_ascii=False, indent=4)

# ---------------------------
# 5️⃣ CHUYỂN EMBEDDING TỪ JSON SANG NUMPY ARRAY
# ---------------------------
questions_list = list(embedding_data.keys())
embeddings_list = [embedding_data[q] for q in questions_list]
embeddings_array = np.array(embeddings_list, dtype="float32")

# ---------------------------
# 6️⃣ KIỂM TRA KÍCH THƯỚC EMBEDDING
# ---------------------------
if embeddings_array.shape[1] != 768:
    print(f"❌ Lỗi: Mô hình e5-base yêu cầu embedding 768 chiều, nhưng dữ liệu có {embeddings_array.shape[1]} chiều!")
    exit()

# ---------------------------
# 7️⃣ CHUẨN HÓA VECTORS VÀ XÂY DỰNG FAISS INDEX
# ---------------------------
faiss.normalize_L2(embeddings_array)
d = 768  # E5-base có 768 chiều
index = faiss.IndexFlatIP(d)  # Dùng inner product (IP) vì đã normalize
index.add(embeddings_array)  # Thêm embedding vào FAISS

# Mapping chỉ số FAISS sang câu hỏi
index_to_question = {i: questions_list[i] for i in range(len(questions_list))}

# ---------------------------
# 8️⃣ HÀM XỬ LÝ TRUY VẤN Q&A SỬ DỤNG FAISS
# ---------------------------
def answer_query_faiss(user_query, similarity_threshold=0.2):
    """Xử lý truy vấn từ người dùng, hỗ trợ cả câu có dấu và không dấu."""
    user_query_no_accent = unidecode(user_query)  # Loại bỏ dấu tiếng Việt
    
    # Embed cả câu hỏi gốc và không dấu
    query_emb = get_embedding_batch([user_query, user_query_no_accent])  

    k = 1  # Lấy kết quả tốt nhất
    distances, indices = index.search(query_emb, k)

    best_score = max(distances[0][0], distances[1][0])  # Chọn điểm số cao nhất
    best_index = indices[0][0] if distances[0][0] > distances[1][0] else indices[1][0]

    if best_score < similarity_threshold:
        return "Chúng tôi chưa hiểu câu hỏi của bạn.", None, None

    best_question = index_to_question[best_index]
    
    for key, value in qa_data.items():
        if best_question in value:
            return key, best_question, best_score  # Trả về câu trả lời, câu hỏi tìm thấy và độ tương đồng
    
    return "Câu hỏi không khớp với dữ liệu, vui lòng thử lại!", None, None

# ---------------------------
# 9️⃣ STREAMLIT UI
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
