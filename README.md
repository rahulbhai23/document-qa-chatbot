# Document Q&A Chatbot (Generative AI)

A Retrieval-Augmented Generation (RAG) app for asking natural-language questions about PDF and DOCX files. Answers are grounded in your documents and cite their sources.

**Stack:** Python, LangChain, FAISS, Sentence-Transformers, Streamlit, RAGAS, an LLM (Anthropic or OpenAI)

## How it works

1. **Load** PDF/DOCX files and tag each page with its source file.
2. **Chunk** text with `RecursiveCharacterTextSplitter` (1000 chars, 150 overlap).
3. **Embed** chunks locally with `all-MiniLM-L6-v2` and store them in a **FAISS** index.
4. **Retrieve** the top 12 chunks by vector similarity, then **rerank** them with a cross-encoder (`ms-marco-MiniLM-L-6-v2`) and keep the best 4.
5. **Generate** with a strict prompt: answer only from the numbered context, cite passages as `[1]`, `[2]`, and refuse when the answer is not present.
6. **Memory:** follow-up questions are rewritten into standalone questions using recent chat history.

## Run it

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                # then add your API key
streamlit run app.py
```

Upload files in the sidebar, click **Process documents**, then ask questions.

## Measure hallucination with RAGAS

1. Put some documents in a `docs/` folder.
2. Edit `eval_questions.json` with 20-30 questions written from those documents (include a few that are NOT answerable).
3. Run:

```bash
python evaluate.py --docs docs/your_file.pdf --questions eval_questions.json
```

It compares a **baseline** (no reranking, loose prompt) against the **full pipeline** and prints faithfulness, hallucination rate (1 - faithfulness) and the relative reduction. Use the numbers from your own run on your resume.

## Files

| File | Purpose |
|---|---|
| `rag_pipeline.py` | Loading, chunking, FAISS, reranking, prompts, answer generation |
| `app.py` | Streamlit interface |
| `evaluate.py` | RAGAS evaluation, baseline vs. full pipeline |
| `eval_questions.json` | Your evaluation questions |

## Limitations

- Scanned (image-only) PDFs need OCR first.
- The index lives in memory and resets when the app restarts.
