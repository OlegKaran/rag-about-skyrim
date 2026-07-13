"""
app.py
------
Gradio-интерфейс для RAG-системы по лору Oblivion (UESP.net).

Запуск:
    python app.py

Перед первым запуском:
1. Скопируй .env.example в .env и заполни QDRANT_URL / QDRANT_API_KEY.
2. Убедись, что коллекция в Qdrant уже создана (см. ingest.py, если ещё нет).
3. Убедись, что Ollama запущена и модель скачана:
       ollama pull qwen2.5-coder:7b
"""

import traceback
from typing import List, Tuple

import gradio as gr

from rag_core import DEFAULT_RERANK_TOP_N, DEFAULT_RETRIEVE_K, OLLAMA_MODEL, RAGPipeline, RerankingRetriever

EXAMPLE_QUESTIONS = [
    "Кто такой Мартин Септим?",
    "Что произошло во время кризиса Обливиона?",
    "Расскажи про гильдию магов в Oblivion",
    "Какие фракции есть в Сиродиле?",
]


def load_pipeline():
    """Загружает пайплайн один раз при старте приложения.
    Если что-то не настроено (.env, Ollama, Qdrant) — приложение всё равно
    запустится, но покажет понятную ошибку в интерфейсе вместо падения."""
    try:
        retriever = RerankingRetriever()
        return RAGPipeline(retriever=retriever), None
    except Exception as exc:  # noqa: BLE001 — здесь осознанно ловим всё
        return None, str(exc)


PIPELINE, INIT_ERROR = load_pipeline()


def format_sources_markdown(scored_docs: List[Tuple[object, float]]) -> str:
    if not scored_docs:
        return "_Источники не найдены._"
    lines = ["### Источники, использованные для ответа"]
    for i, (doc, score) in enumerate(scored_docs, start=1):
        title = doc.metadata.get("section_title") or doc.metadata.get("title") or "без названия"
        excerpt = doc.page_content[:350].strip().replace("\n", " ")
        lines.append(f"**{i}. {title}**  \nscore реранкера: `{score:.3f}`\n\n> {excerpt}…")
    return "\n\n---\n\n".join(lines)


def respond(message, history, top_n, retrieve_k, temperature):
    if not message or not message.strip():
        return history, "", gr.update()

    history = history + [{"role": "user", "content": message}]

    if PIPELINE is None:
        error_text = (
            "⚠️ Пайплайн не инициализирован.\n\n"
            f"Причина: {INIT_ERROR}\n\n"
            "Проверь файл .env и что Ollama/Qdrant доступны, затем перезапусти приложение."
        )
        history = history + [{"role": "assistant", "content": error_text}]
        return history, "", ""

    try:
        answer, scored_docs = PIPELINE.answer(
            message,
            retrieve_k=int(retrieve_k),
            top_n=int(top_n),
            temperature=float(temperature),
        )
        sources_md = format_sources_markdown(scored_docs)
    except Exception:
        answer = (
            "⚠️ Не получилось получить ответ. Проверь, что Ollama запущена "
            "(`ollama serve`) и модель скачана, а Qdrant доступен.\n\n"
            f"Техническая деталь:\n```\n{traceback.format_exc()[-800:]}\n```"
        )
        sources_md = ""

    history = history + [{"role": "assistant", "content": answer}]
    return history, "", sources_md


def clear_chat():
    return [], "", ""


with gr.Blocks(title="Oblivion Lore Assistant") as demo:
    gr.Markdown(
        "# 🏛️ Oblivion Lore Assistant\n"
        "RAG по вики-статьям UESP.net о The Elder Scrolls IV: Oblivion. "
        f"Ретрив → реранк (bge-reranker-v2-m3) → генерация ответа ({OLLAMA_MODEL})."
    )

    if INIT_ERROR:
        gr.Markdown(f"> ⚠️ **Внимание:** пайплайн не загрузился при старте: `{INIT_ERROR}`")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=480, type="messages", label="Диалог")
            msg = gr.Textbox(
                placeholder="Спроси что-нибудь про лор Oblivion...",
                label="Вопрос",
                lines=2,
            )
            with gr.Row():
                submit_btn = gr.Button("Спросить", variant="primary")
                clear_btn = gr.Button("Очистить диалог")
            gr.Examples(examples=EXAMPLE_QUESTIONS, inputs=msg, label="Примеры вопросов")
            sources_box = gr.Markdown(label="Источники")

        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Настройки поиска")
            top_n_slider = gr.Slider(
                1, 10, value=DEFAULT_RERANK_TOP_N, step=1,
                label="Rerank top_n (сколько фрагментов уходит в промпт)",
            )
            retrieve_k_slider = gr.Slider(
                5, 40, value=DEFAULT_RETRIEVE_K, step=1,
                label="Retrieve K (сколько кандидатов достаём из Qdrant до реранка)",
            )
            temperature_slider = gr.Slider(
                0.0, 1.0, value=0.0, step=0.1,
                label="Temperature генерации",
            )

    submit_btn.click(
        respond,
        inputs=[msg, chatbot, top_n_slider, retrieve_k_slider, temperature_slider],
        outputs=[chatbot, msg, sources_box],
    )
    msg.submit(
        respond,
        inputs=[msg, chatbot, top_n_slider, retrieve_k_slider, temperature_slider],
        outputs=[chatbot, msg, sources_box],
    )
    clear_btn.click(clear_chat, inputs=None, outputs=[chatbot, msg, sources_box])


if __name__ == "__main__":
    # queue() нужен, чтобы запросы к LLM не блокировали друг друга намертво
    # при нескольких одновременных пользователях.
    demo.queue().launch(
        server_name="0.0.0.0",  # доступно не только с localhost (важно для деплоя на сервере/в Docker)
        server_port=7860,
        share=False,  # поставь True, если хочешь временную публичную ссылку через Gradio
    )
