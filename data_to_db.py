import json
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()
qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
input_file_path = "rag_data/Oblivion_Cleaned-2.jsonl"
print("Загрузка данных:")
raw_documents = []
with open(input_file_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        metadata = data["metadata"]
        for section_title, section_text in data["clean_text"].items():
            chunk_metadata = metadata.copy()
            chunk_metadata["section_title"] = section_title
            doc = Document(page_content=section_text, metadata=chunk_metadata)
            raw_documents.append(doc)
print(f"Всего создано: {len(raw_documents)} строк")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=768,
    chunk_overlap=96,
    separators=["\n\n", "\n", ". ", " ", ""]
)

documents = text_splitter.split_documents(raw_documents)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "mps"}
)

print("Создание векторов в Qdrant Cloud")
vector_store = QdrantVectorStore.from_documents(
    [documents[0]],
    embeddings,
    url=qdrant_url,
    api_key=qdrant_api_key,
    collection_name="oblivion_lore",
    force_recreate=True
)

batch_size = 400

print(f"Начинаем загрузку {len(documents)} чанков батчами по {batch_size}...")

for i in range(1, len(documents), batch_size):
    batch = documents[i : i + batch_size]
    vector_store.add_documents(batch)
    print(f"Загружено {min(i + batch_size, len(documents))} / {len(documents)} чанков")

print("Загрузка успешно завершена!")