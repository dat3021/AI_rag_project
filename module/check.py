from langchain_chroma import Chroma
from embedding import embedding


# Kết nối lại vào DB đã lưu ở thư mục cục bộ
db = Chroma(
    persist_directory="./vector_db",
    embedding_function=embedding # Phải truyền đúng model embedding lúc lưu
)
#----------------------------------------------------------------------------------------
# db.delete_collection()
# print("Đã xóa sạch bộ lưu trữ (Collection) trong ChromaDB!")
#----------------------------------------------------------------------------------------

# Lấy toàn bộ dữ liệu (mặc định gồm ids, documents, metadatas)
all_data = db.get()

total_records = len(all_data['ids'])
print(f"Tổng số bản ghi trong DB: {total_records}")

# Xem nhanh 5 bản ghi đầu tiên bằng cách slice list
print("\n--- Xem nhanh tối đa 5 bản ghi đầu ---")
limit = min(5, total_records)

for i in range(limit):
    print(f"\n[Bản ghi thứ {i+1}]")
    print(f"ID: {all_data['ids'][i]}")
    print(f"Nội dung văn bản (đoạn đầu): {all_data['documents'][i][:150]}...") # Hiện 150 ký tự đầu
    print(f"Metadata: {all_data['metadatas'][i] if all_data['metadatas'] else 'Không có'}")