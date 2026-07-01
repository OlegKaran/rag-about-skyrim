from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from ragas.testset import TestsetGenerator
from ragas.run_config import RunConfig
import json
import pandas as pd
import time
import os
import random


llm = ChatOllama(
    model="qwen2.5-coder:14b",
    temperature=0.0,
    format="json"
)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "mps"}
)

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
    chunk_size=512,
    chunk_overlap=64
)

documents = text_splitter.split_documents(raw_documents)
random.seed(42)
SAMPLED_DOCS = random.sample(documents, min(100, len(documents)))

# ragas_llm = LangchainLLMWrapper(gemini_llm)
# ragas_embed = LangchainEmbeddingsWrapper(embeddings)


print(f"Создание knowledge graph...")
generator = TestsetGenerator.from_langchain(
    generator_llm=llm,
    critic_llm=llm,
    embeddings=embeddings
)

safe_config = RunConfig(
    max_workers=2,
    max_retries=3,
    max_wait=10
)

TARGET_TOTAL_QUESTIONS = 100
BATCH_SIZE = 20
OUTPUT_FILE = "rag_data/oblivion_golden_set_for_eval.csv"

questions_generated = 0

if os.path.exists(OUTPUT_FILE):
    existing_questions = pd.read_csv(OUTPUT_FILE)
    questions_generated = len(existing_questions)
    print(f"Уже сгенерировано: {questions_generated} вопросов")

print(f"Осталось вопросов: {TARGET_TOTAL_QUESTIONS - questions_generated}")


while questions_generated < TARGET_TOTAL_QUESTIONS:
    testset = generator.generate_with_langchain_docs(
        documents=SAMPLED_DOCS,
        test_size=BATCH_SIZE,
        run_config=safe_config
    )

    batch_df = testset.to_pandas()

    write_header = not os.path.exists(OUTPUT_FILE)
    batch_df.to_csv(OUTPUT_FILE, mode='a', header=write_header, index=False)

    questions_generated += len(batch_df)
    print(f"Успешно сгенерировано и создано {questions_generated} вопросов - ответов")
    time.sleep(15)

print(f"Генерация golden set завершена")




