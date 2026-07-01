from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datasets import Dataset
from ragas.metrics import answer_correctness, context_precision, context_recall, faithfulness
from ragas import evaluate
from ragas.run_config import RunConfig
from dotenv import load_dotenv
import pandas as pd
import os
import json
import itertools
import gc


load_dotenv()
qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'mps'}
    )

eval_llm = ChatOllama(
        model="qwen2.5-coder:14b",
        temperature=0.0
    )

judge_llm = ChatOllama(
        model="qwen2.5-coder:7b",
        temperature=0.0,
        timeout=3600,
        num_ctx=4096,
    )


def generate_rag_answers(retriever, golden_set_df: pd.DataFrame) -> pd.DataFrame:
    system_prompt = (
        "Ты — эксперт по вселенной The Elder Scrolls Oblivion. "
        "Используй следующие фрагменты лора, чтобы ответить на вопрос пользователя. "
        "Если ответа нет в фрагментах, честно скажи, что не знаешь. "
        "Не выдумывай информацию.\n\n"
        "Контекст:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(eval_llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    llm_answers = []

    for i, row in golden_set_df.iterrows():
        question = row["user_input"]
        ideal_reference = row["reference"]
        response = rag_chain.invoke({"input": question})
        contexts = [doc.page_content for doc in response["context"]]
        llm_answers.append({
            "question": question,
            "answer": response["answer"],
            "contexts": contexts,
            "ground_truth": ideal_reference
        })
    return pd.DataFrame(llm_answers)


def evaluate_rag_metrics(results: pd.DataFrame) -> dict:
    data_dict = {
        "question": results["question"].tolist(),
        "answer": results["answer"].tolist(),
        "contexts": results["contexts"].tolist(),
        "ground_truth": results["ground_truth"].tolist()
    }
    dataset = Dataset.from_dict(data_dict)

    metrics=[context_precision, context_recall]
    config = RunConfig(timeout=3600, max_workers=1, max_retries=3)

    metrics_results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=embeddings,
        run_config=config,
        raise_exceptions=True
    )
    return metrics_results


def rag_tuning_pipeline(input_file_path: str, golden_set_path: str):
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

    full_golden_set_df = pd.read_csv(golden_set_path)
    golden_set_df = full_golden_set_df.sample(n=20, random_state=42)
    param_grid = {
        "chunk_size": [768],
        "chunk_overlap": [96],
        "top_k": [3]
    }

    keys, values = zip(*param_grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    best_score = 0
    best_params = {}
    all_results = []

    for idx, params in enumerate(experiments):
        print(f"\n--- Эксперимент {idx + 1}/{len(experiments)} ---")
        print(f"Параметры: {params}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=params["chunk_size"],
            chunk_overlap=params["chunk_overlap"],
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        documents = text_splitter.split_documents(raw_documents)

        print("Создание векторов в qdrant базе in-memory")
        vector_store = QdrantVectorStore.from_documents(
            documents=[documents[0]],
            embedding=embeddings,
            location=":memory:",
            collection_name="rag_eval"
        )

        batch_size = 400

        print(f"Начинаем загрузку {len(documents)} чанков батчами по {batch_size}...")

        for i in range(1, len(documents), batch_size):
            batch = documents[i: i + batch_size]
            vector_store.add_documents(batch)
            print(f"Загружено {min(i + batch_size, len(documents))} / {len(documents)} чанков")

        print("Загрузка успешно завершена!")

        retriever = vector_store.as_retriever(search_kwargs={"k": params["top_k"]})

        results_df = generate_rag_answers(retriever, golden_set_df)
        metrics_result = evaluate_rag_metrics(results_df)
        avg_score = metrics_result['context_recall']
        print(f"Результат Recall: {avg_score:.4f}")
        checkpoint_data = {"params": params, "metrics": metrics_result}
        with open("rag_data/tuning_checkpoints.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(checkpoint_data) + "\n")

        all_results.append({
            "params": params,
            "metrics": metrics_result
        })

        if avg_score > best_score:
            best_score = avg_score
            best_params = params
        del vector_store
        del retriever
        del documents
        del results_df
        gc.collect()
        print("Память очищена")
    print(f"Лучшие параметры: {best_params}")
    print(f"С лучшим скором Context Recall: {best_score:.4f}")



if __name__ == "__main__":
    GOLDEN_SET_PATH = "rag_data/oblivion_golden_set.csv"
    # VAL_DATA_PATH = "rag_data/val_data.csv"
    # METRICS = "rag_data/metrics"
    INPUT_JSON_FILE_PATH = "rag_data/Oblivion_Cleaned-2.jsonl"
    INPUT_UPDATED_JSON_FILE_PATH = "rag_data/Oblivion_Cleaned_fixed.jsonl"
    # rag_tuning_pipeline(input_file_path=INPUT_JSON_FILE_PATH, golden_set_path=GOLDEN_SET_PATH)
    rag_tuning_pipeline(input_file_path=INPUT_UPDATED_JSON_FILE_PATH, golden_set_path=GOLDEN_SET_PATH)


