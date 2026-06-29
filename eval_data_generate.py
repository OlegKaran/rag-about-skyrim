from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from datasets import Dataset
from ragas.metrics import answer_correctness, context_precision, context_recall, faithfulness
from ragas import evaluate
from ragas.run_config import RunConfig
from dotenv import load_dotenv
import ast
import pandas as pd
import os


load_dotenv()
qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
def generate_rag_data(golden_set_path: str, output_csv_path: str):
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'mps'}
    )

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name="oblivion_lore"
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    eval_llm = ChatOllama(
        model="qwen2.5-coder:14b",
        temperature=0.0
    )

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

    golden_set_data = pd.read_csv(golden_set_path)
    llm_answers = []

    for i, row in golden_set_data.iterrows():
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
        if len(llm_answers) % 10 == 0:
            print(f"Сгенерировано ответов: {len(llm_answers)} / {len(golden_set_data['user_input'])}")

    results = pd.DataFrame(llm_answers)
    results.to_csv(output_csv_path, index=False)
    print(f"Всего сгенерировано: {len(results)} ответов. Они сохранены в {output_csv_path}")


def evaluate_rag(val_data_path: str, output_metrics_path: str):
    val_data_df = pd.read_csv(val_data_path)
    val_data_df["contexts"] = val_data_df["contexts"].apply(ast.literal_eval)
    data_dict = {
        "question": val_data_df["question"].tolist(),
        "answer": val_data_df["answer"].tolist(),
        "contexts": val_data_df["contexts"].tolist(),
        "ground_truth": val_data_df["ground_truth"].tolist()
    }

    dataset = Dataset.from_dict(data_dict)

    judge_llm = ChatOllama(
        model="qwen2.5-coder:14b",
        temperature=0.0,
        timeout=3600,
        num_ctx=3000
    )
    judge_embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'mps'}
    )

    metrics=[answer_correctness, context_precision, context_recall, faithfulness]
    config = RunConfig(timeout=3600, max_workers=1, max_retries=2)
    results = evaluate(
        dataset=dataset.select(range(3)),
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=config,
        raise_exceptions=True
    )

    print(results)
    results_df = results.to_pandas()
    results_df.to_csv(output_metrics_path, index=False)
    print(f"Метрики сохранены в {output_metrics_path}")


GOLDEN_SET_PATH = "rag_data/oblivion_golden_set.csv"
VAL_DATA_PATH = "rag_data/val_data.csv"
METRICS = "rag_data/metrics"
# generate_rag_data(golden_set_path=GOLDEN_SET_PATH, output_csv_path=VAL_DATA_PATH)
evaluate_rag(val_data_path=VAL_DATA_PATH, output_metrics_path=METRICS)


