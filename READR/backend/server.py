"""AI Document Processing & Insight Extractor.

Sections in this file:
1. Imports and app setup: loads dependencies, creates the FastAPI app, and mounts the UI folder.
2. Utility helpers: extracts text, performs OCR fallback, chunks/indexes documents, and talks to Ollama.
3. API models and endpoints: handles upload, model switching, hybrid retrieval, and chat history reset.
4. Local server entrypoint: starts the FastAPI app with uvicorn when running `python server.py`.

# Run these before starting server:
# ollama pull llama3.1:8b-instruct-q4_K_M
# ollama pull nomic-embed-text
# ollama pull qwen3-vl:8b  (only if using OCR)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

import chromadb
import fitz
import ollama
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rank_bm25 import BM25Okapi


MODEL_NAME = "llama3.1:8b-instruct-q4_K_M"
EMBED_MODEL = "nomic-embed-text"
VISION_MODEL = "qwen3-vl:8b"
CURRENT_MODEL = MODEL_NAME
RAG_MODE = True
ALLOWED_MODELS = {MODEL_NAME, VISION_MODEL}
BASE_DIR = Path(__file__).resolve().parent
UI_DIR = BASE_DIR.parent / "ui"
MAX_CONTEXT_CHARS = 10000
SUMMARY_CONTEXT_CHARS = 6000
CHUNK_SIZE = 1200
MIN_CHUNK_SIZE = 80
MAX_HISTORY_MESSAGES = 6
DEBUG_OLLAMA_PROMPTS = True
MAX_DEBUG_PROMPT_LOGS = 80

CURRENT_DOC_ID: str | None = None
CURRENT_DOC_TEXT = ""
DOCUMENT_CACHE: dict[str, dict[str, Any]] = {}
DOCUMENT_FULL_TEXT: dict[str, str] = {}
CHAT_HISTORY: dict[str, list[dict[str, str]]] = {}
DEBUG_PROMPT_LOGS: list[dict[str, Any]] = []

chroma_client = chromadb.PersistentClient(path=str(BASE_DIR / "chroma_db"))
collection = chroma_client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"},
)

app = FastAPI(title="AI Document Processing & Insight Extractor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def normalize_for_search(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def clamp_text(text: str, limit: int = MAX_CONTEXT_CHARS) -> str:
    return text.strip()[:limit]


def add_debug_log(entry_type: str, title: str, content: str, **extra: Any) -> None:
    if not DEBUG_OLLAMA_PROMPTS:
        return

    DEBUG_PROMPT_LOGS.append(
        {
            "type": entry_type,
            "title": title,
            "content": content,
            "extra": extra,
        }
    )
    if len(DEBUG_PROMPT_LOGS) > MAX_DEBUG_PROMPT_LOGS:
        del DEBUG_PROMPT_LOGS[:-MAX_DEBUG_PROMPT_LOGS]


def debug_print_ollama_chat(model: str, messages: list[dict[str, Any]], format: str | None = None) -> None:
    if not DEBUG_OLLAMA_PROMPTS:
        return

    parts = [f"Model: {model}"]
    if format:
        parts.append(f"Format: {format}")
    for index, message in enumerate(messages, start=1):
        part_lines = [
            f"Message {index}",
            f"Role: {message.get('role', 'unknown')}",
        ]
        if message.get("images"):
            part_lines.append(f"Images: {len(message['images'])} attached")
        part_lines.append("Content:")
        part_lines.append(str(message.get("content", "")))
        parts.append("\n".join(part_lines))
    add_debug_log("chat", "Ollama Chat Request", "\n\n".join(parts), model=model, format=format)

    print("\n" + "=" * 80)
    print("OLLAMA CHAT REQUEST")
    print(f"Model: {model}")
    if format:
        print(f"Format: {format}")
    for index, message in enumerate(messages, start=1):
        print("-" * 80)
        print(f"Message {index}")
        print(f"Role: {message.get('role', 'unknown')}")
        if message.get("images"):
            print(f"Images: {len(message['images'])} attached")
        print("Content:")
        print(message.get("content", ""))
    print("=" * 80 + "\n")


def debug_print_ollama_embed(model: str, texts: list[str]) -> None:
    if not DEBUG_OLLAMA_PROMPTS:
        return

    parts = [f"Model: {model}", f"Inputs: {len(texts)}"]
    for index, text in enumerate(texts, start=1):
        parts.append(f"Input {index}\n{text}")
    add_debug_log("embed", "Ollama Embed Request", "\n\n".join(parts), model=model, count=len(texts))

    print("\n" + "=" * 80)
    print("OLLAMA EMBED REQUEST")
    print(f"Model: {model}")
    print(f"Inputs: {len(texts)}")
    for index, text in enumerate(texts, start=1):
        print("-" * 80)
        print(f"Input {index}")
        print(text)
    print("=" * 80 + "\n")


def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as document:
            return "\n".join(page.get_text("text") for page in document).strip()
    except Exception as exc:  # pragma: no cover - defensive error path
        raise HTTPException(status_code=400, detail=f"Failed to read PDF: {exc}") from exc


def render_pdf_pages(file_bytes: bytes) -> list[bytes]:
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as document:
            return [page.get_pixmap().tobytes("png") for page in document]
    except Exception as exc:  # pragma: no cover - defensive error path
        raise HTTPException(status_code=400, detail=f"Failed to render PDF pages: {exc}") from exc


def call_ollama_chat(
    *,
    model: str,
    messages: list[dict[str, Any]],
    format: str | None = None,
) -> str:
    debug_print_ollama_chat(model=model, messages=messages, format=format)
    try:
        response = ollama.chat(model=model, messages=messages, format=format)
    except Exception as exc:  # pragma: no cover - depends on local Ollama availability
        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to contact Ollama at http://localhost:11434. "
                "Make sure Ollama is running and the required models are installed."
            ),
        ) from exc
    return response["message"]["content"].strip()


def call_ollama_json(prompt: str) -> dict[str, Any]:
    content = call_ollama_chat(
        model=CURRENT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You extract grounded insights from documents. "
                    "Return valid JSON only, with no markdown fences or extra commentary."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        format="json",
    )
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="Model returned invalid JSON.") from exc


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    debug_print_ollama_embed(model=EMBED_MODEL, texts=texts)
    try:
        response = ollama.embed(model=EMBED_MODEL, input=texts)
    except Exception as exc:  # pragma: no cover - depends on local Ollama availability
        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to generate embeddings with nomic-embed-text. "
                "Run `ollama pull nomic-embed-text` and make sure Ollama is running. "
                f"Original error: {exc}"
            ),
        ) from exc
    embeddings = response.get("embeddings", [])
    if len(embeddings) != len(texts):
        raise HTTPException(status_code=500, detail="Embedding model returned invalid vectors.")
    return embeddings


def embed_text(text: str) -> list[float]:
    embeddings = embed_texts([text])
    return embeddings[0] if embeddings else []


def rewrite_question(question: str) -> str:
    prompt = (
        "Rewrite this question as a clear full question.\n"
        "IMPORTANT RULES for academic grade documents:\n"
        "- 'O' always means the letter GRADE Outstanding (90-100%)\n"
        "- Never interpret O as an abbreviation or acronym\n"
        "- 'first time' means chronologically earliest semester\n"
        "- Preserve the exact meaning, only expand if truly needed\n\n"
        f"Question: {question}\n"
        "Rewritten question:"
    )
    return call_ollama_chat(model=CURRENT_MODEL, messages=[{"role": "user", "content": prompt}])


def classify_query(question: str) -> str:
    # Check for temporal keywords first, before calling LLM
    if any(word in question.lower() for word in ["first", "last", "when", "earliest", "latest"]):
        return "TEMPORAL"
    
    prompt = f"""
Classify this question into one category.
Reply with ONLY the category name, nothing else.

Categories:
- SINGLE: answer found in one place (what is X, what grade did I get in Y)
- COMPARATIVE: needs comparing across sections (which is best, which semester had most O grades)
- TEMPORAL: needs chronological order (first time, last time, improved over time)
- LISTING: needs all matches (list all subjects with O, show all A+ grades)
- GRADING_RULE: about grading system (what is grade point for O, what is SGPA)

Question: {question}
Category:
""".strip()
    classification = call_ollama_chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}]).strip()
    return classification if classification in {"SINGLE", "COMPARATIVE", "TEMPORAL", "LISTING", "GRADING_RULE"} else "SINGLE"


def score_chunk_relevance(question: str, chunk: str) -> float:
    prompt = (
        "Score how relevant this text is to the question on a scale 0-10. "
        "Reply with just the number.\n"
        f"Question: {question}\n"
        f"Text: {chunk[:300]}"
    )
    raw_score = call_ollama_chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
    match = re.search(r"\d+(?:\.\d+)?", raw_score)
    return float(match.group(0)) if match else 0.0


def summarize_history(messages: list[dict[str, str]]) -> dict[str, str]:
    prompt = "Summarise these exchanges in 2 sentences: " + " ".join(
        f"{item['role']}: {item['content']}" for item in messages
    )
    summary = call_ollama_chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
    return {"role": "system", "content": summary}


def trim_history(session_id: str) -> list[dict[str, str]]:
    history = CHAT_HISTORY.get(session_id, [])
    if len(history) > MAX_HISTORY_MESSAGES:
        summary = summarize_history(history[:4])
        history = [summary] + history[4:]
    if len(history) > MAX_HISTORY_MESSAGES:
        history = history[-MAX_HISTORY_MESSAGES:]
    CHAT_HISTORY[session_id] = history
    return history


def extract_text_with_qwen_ocr(page_images: list[bytes]) -> str:
    page_texts = []
    for page_image in page_images:
        text = call_ollama_chat(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Extract all text from this image exactly as written. "
                        "Return only the text, nothing else."
                    ),
                    "images": [page_image],
                }
            ],
        )
        if text.strip():
            page_texts.append(text.strip())
    return "\n\n".join(page_texts).strip()


def extract_text(filename: str, file_bytes: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".txt":
        try:
            return file_bytes.decode("utf-8").strip()
        except UnicodeDecodeError:
            return file_bytes.decode("latin-1").strip()
    if suffix != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")

    page_images = render_pdf_pages(file_bytes)
    pdf_text = extract_text_from_pdf(file_bytes)
    if CURRENT_MODEL == VISION_MODEL:
        return extract_text_with_qwen_ocr(page_images)
    if len(pdf_text.strip()) < 50:
        return extract_text_with_qwen_ocr(page_images)
    return pdf_text


def semantic_chunk(text: str, max_size: int = CHUNK_SIZE) -> list[str]:
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) < max_size:
            current = f"{current}\n\n{para}".strip() if current else para
            continue

        if current:
            chunks.append(current.strip())

        if len(para) <= max_size:
            current = para
            continue

        remainder = para
        for separator in ("\n", ". ", ", ", " "):
            if len(remainder) <= max_size:
                break
            pieces = []
            working = ""
            for part in remainder.split(separator):
                candidate = f"{working}{separator}{part}".strip(separator) if working else part
                if len(candidate) < max_size:
                    working = candidate
                else:
                    if working:
                        pieces.append(working.strip())
                    working = part
            if working:
                pieces.append(working.strip())
            if pieces and all(len(piece) <= max_size for piece in pieces):
                chunks.extend(piece for piece in pieces[:-1] if len(piece) > MIN_CHUNK_SIZE)
                current = pieces[-1]
                remainder = current
                break
        else:
            for start in range(0, len(para), max_size):
                piece = para[start : start + max_size].strip()
                if len(piece) > MIN_CHUNK_SIZE:
                    chunks.append(piece)
            current = ""

    if current:
        chunks.append(current.strip())

    return [chunk for chunk in chunks if len(chunk) > MIN_CHUNK_SIZE]


def detect_semester(chunk: str) -> int | None:
    match = re.search(r"(semester|sem)\s*[:\-]?\s*(\d+)", chunk, re.IGNORECASE)
    return int(match.group(2)) if match else None


def chunk_metadata(chunk: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {"type": "general"}
    semester = detect_semester(chunk)
    if semester is not None:
        metadata["semester"] = semester
    if "F1=Fail" in chunk or "F2=Fail" in chunk:
        metadata["type"] = "grade_scale"
    elif "SGPA" in chunk and "Subject Code" in chunk:
        metadata["type"] = "semester_data"
    return metadata


def build_document_index(text: str, doc_id: str | None = None) -> tuple[str, list[dict[str, Any]]]:
    doc_id = doc_id or str(uuid4())
    chunks = semantic_chunk(text)
    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="Could not create meaningful chunks from the uploaded document.",
        )
    embeddings = embed_texts(chunks)
    metadatas = []
    for index, chunk in enumerate(chunks):
        metadata = chunk_metadata(chunk)
        metadata.update({"doc_id": doc_id, "chunk_index": index})
        metadatas.append(metadata)

    ids = [f"{doc_id}_chunk_{index}" for index in range(len(chunks))]
    try:
        collection.add(documents=chunks, embeddings=embeddings, ids=ids, metadatas=metadatas)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to store document in ChromaDB: {exc}") from exc

    tokenized = [normalize_for_search(chunk).split() for chunk in chunks]
    DOCUMENT_CACHE[doc_id] = {
        "chunks": chunks,
        "metadatas": metadatas,
        "bm25": BM25Okapi(tokenized) if tokenized else None,
    }
    return doc_id, [{"chunk": chunk, "metadata": metadata} for chunk, metadata in zip(chunks, metadatas)]


def bm25_top_chunks(doc_id: str, question: str, top_k: int = 5) -> list[tuple[int, str]]:
    cache = DOCUMENT_CACHE.get(doc_id, {})
    bm25 = cache.get("bm25")
    chunks = cache.get("chunks", [])
    if not bm25 or not chunks:
        return []
    scores = bm25.get_scores(normalize_for_search(question).split())
    ranked_indexes = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:top_k]
    return [(index, chunks[index]) for index in ranked_indexes if scores[index] > 0]


def chroma_top_chunks(doc_id: str, question_embedding: list[float], top_k: int = 5) -> list[str]:
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
        where={"doc_id": doc_id},
    )
    return results.get("documents", [[]])[0]


def filter_retrievable_chunks(doc_id: str, chunks: list[str], classification: str) -> list[str]:
    cache = DOCUMENT_CACHE.get(doc_id, {})
    metadatas = cache.get("metadatas", [])
    lookup = {cache["chunks"][index]: metadatas[index] for index in range(len(cache.get("chunks", [])))}
    filtered = []
    for chunk in chunks:
        metadata = lookup.get(chunk, {})
        if metadata.get("type") == "grade_scale" and classification != "GRADING_RULE":
            continue
        filtered.append(chunk)
    return filtered


def retrieve_for_temporal(doc_id: str) -> list[str]:
    cache = DOCUMENT_CACHE.get(doc_id, {})
    paired = []
    for chunk, metadata in zip(cache.get("chunks", []), cache.get("metadatas", [])):
        semester = metadata.get("semester")
        if semester is not None:
            paired.append((semester, chunk))
    paired.sort(key=lambda item: item[0])
    return [chunk for _, chunk in paired]


def retrieve_for_grading_rule(doc_id: str) -> list[str]:
    cache = DOCUMENT_CACHE.get(doc_id, {})
    return [
        chunk
        for chunk, metadata in zip(cache.get("chunks", []), cache.get("metadatas", []))
        if metadata.get("type") == "grade_scale"
    ]


def retrieve_context(doc_id: str, original_question: str, rewritten_question: str, classification: str) -> tuple[str, dict[str, Any]]:
    """
    Retrieve context and return both the context and detailed retrieval info for logging.
    Returns: (context_text, debug_info_dict)
    """
    debug_info = {
        "embedding_length": 0,
        "vector_chunks": [],
        "keyword_chunks": [],
        "merged_chunks": [],
        "final_chunks": [],
    }

    if classification == "TEMPORAL":
        # For TEMPORAL queries: get ALL chunks from ChromaDB, sorted chronologically
        results = collection.get(where={"doc_id": doc_id})
        
        if not results or not results.get("documents"):
            final_chunks = []
        else:
            # Sort by chunk_index metadata (chronological order)
            all_chunks = sorted(
                zip(results["documents"], results["metadatas"]),
                key=lambda x: x[1].get("chunk_index", 0)
            )
            
            # Filter out grade scale chunks
            semester_chunks = [
                chunk for chunk, meta in all_chunks
                if "F1=Fail" not in chunk and "F2=Fail" not in chunk
            ]
            
            # Send ALL semester chunks as context
            final_chunks = semester_chunks
        
        debug_info["final_chunks"] = [(None, chunk) for chunk in final_chunks]
    else:
        question_embedding = embed_text(rewritten_question)
        cache = DOCUMENT_CACHE.get(doc_id, {})
        debug_info["embedding_length"] = len(question_embedding)
        
        if classification == "GRADING_RULE":
            merged = retrieve_for_grading_rule(doc_id)
        elif classification == "LISTING":
            merged = [chunk for _, chunk in bm25_top_chunks(doc_id, rewritten_question, top_k=len(cache.get("chunks", [])))]
        else:
            vector_limit = 8 if classification == "COMPARATIVE" else 3
            vector_results = chroma_top_chunks(doc_id, question_embedding, top_k=max(vector_limit, 5))
            keyword_results = bm25_top_chunks(doc_id, rewritten_question, top_k=5)
            
            debug_info["vector_chunks"] = vector_results
            debug_info["keyword_chunks"] = [(str(idx), chunk) for idx, chunk in keyword_results]
            
            merged = []
            for chunk in vector_results + [chunk for _, chunk in keyword_results]:
                if chunk not in merged:
                    merged.append(chunk)

        merged = filter_retrievable_chunks(doc_id, merged, classification)
        debug_info["merged_chunks"] = merged
        
        scored = []
        for chunk in merged:
            scored.append((score_chunk_relevance(rewritten_question, chunk), chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        final_chunks = [chunk for _, chunk in scored[:3]]
        debug_info["final_chunks"] = [(score, chunk) for score, chunk in scored[:3]]
    
    return clamp_text("\n\n".join(final_chunks)), debug_info


class QueryRequest(BaseModel):
    query: str
    context: str | None = None
    session_id: str = "default"


class ModelRequest(BaseModel):
    model: str


class ModeRequest(BaseModel):
    mode: str


class NewChatRequest(BaseModel):
    session_id: str = "default"


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> dict[str, Any]:
    global CURRENT_DOC_ID, CURRENT_DOC_TEXT

    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name.")

    file_bytes = await file.read()
    document_text = extract_text(file.filename, file_bytes)
    if not document_text:
        raise HTTPException(status_code=400, detail="The uploaded document is empty.")

    doc_id = str(uuid4())
    CURRENT_DOC_TEXT = document_text
    CURRENT_DOC_ID = doc_id
    DOCUMENT_FULL_TEXT[doc_id] = document_text

    print("\n" + "=" * 80)
    print("UPLOAD DOCUMENT")
    print("=" * 80)
    print(f"Filename: {file.filename}")
    print(f"Doc ID: {doc_id}")
    print(f"Mode: {'RAG' if RAG_MODE else 'DIRECT'}")
    
    print("\n=== EXTRACTED TEXT PREVIEW ===")
    print(f"Total characters: {len(document_text)}")
    print(f"Total words: {len(document_text.split())}")
    print(f"First 500 characters:\n{document_text[:500]}")
    
    if RAG_MODE:
        # Build index and get detailed info
        chunks = semantic_chunk(document_text)
        print(f"\n=== CHUNKS ===")
        print(f"Number of chunks created: {len(chunks)}")
        if chunks:
            print(f"\nFirst chunk:\n{chunks[0]}\n")
            print(f"Last chunk:\n{chunks[-1]}\n")
        
        # Get embeddings
        embeddings = embed_texts(chunks)
        print(f"=== EMBEDDINGS ===")
        if embeddings:
            print(f"Embedding vector length: {len(embeddings[0])}")
            print(f"Total embeddings generated: {len(embeddings)}")
        
        # Build index and get ChromaDB count
        build_document_index(document_text, doc_id=doc_id)
        chroma_count = collection.count()
        print(f"\n=== CHROMADB ===")
        print(f"Collection count after insert: {chroma_count}")
        
        # Additional debug logging
        print("=== EMBEDDING MODEL USED ===", EMBED_MODEL)
        print("=== CHROMADB COUNT ===", collection.count())
        print("=== CHUNKS CREATED ===", len(chunks))
        if chunks:
            print("=== FIRST CHUNK ===", chunks[0][:300])
    else:
        print(f"\nDirect mode: Skipping RAG index build")
        DOCUMENT_CACHE.pop(doc_id, None)
    
    print("=" * 80 + "\n")

    context = clamp_text(document_text, SUMMARY_CONTEXT_CHARS)
    prompt = f"""
Analyze the document below and return JSON with this exact shape:
{{
  "summary": "short paragraph summary",
  "entities": {{
    "names": ["..."],
    "orgs": ["..."],
    "dates": ["..."],
    "values": ["..."]
  }},
  "insights": ["...", "..."]
}}

Rules:
- Stay grounded in the document.
- Keep the summary concise.
- Include only meaningful entities.
- Insights must be short bullet-style statements.

Document:
{context}
""".strip()

    result = call_ollama_json(prompt)
    return {
        "summary": result.get("summary", ""),
        "entities": {
            "names": result.get("entities", {}).get("names", []),
            "orgs": result.get("entities", {}).get("orgs", []),
            "dates": result.get("entities", {}).get("dates", []),
            "values": result.get("entities", {}).get("values", []),
        },
        "insights": result.get("insights", []),
        "context": context,
        "word_count": len(document_text.split()),
        "mode": "rag" if RAG_MODE else "direct",
    }


@app.post("/query")
async def query_document(payload: QueryRequest) -> dict[str, str]:
    original_question = payload.query.strip()
    if not original_question:
        raise HTTPException(status_code=400, detail="Query is required.")
    if not CURRENT_DOC_ID or CURRENT_DOC_ID not in DOCUMENT_FULL_TEXT:
        raise HTTPException(status_code=400, detail="Upload a document before asking questions.")

    print("\n" + "=" * 80)
    print("QUERY DOCUMENT")
    print("=" * 80)

    if RAG_MODE:
        if CURRENT_DOC_ID not in DOCUMENT_CACHE:
            build_document_index(DOCUMENT_FULL_TEXT[CURRENT_DOC_ID], doc_id=CURRENT_DOC_ID)

        rewritten_question = rewrite_question(original_question)
        classification = classify_query(rewritten_question)
        context, retrieval_debug = retrieve_context(CURRENT_DOC_ID, original_question, rewritten_question, classification)

        print(f"\nOriginal question: {original_question}")
        print(f"Rewritten question: {rewritten_question}")
        print(f"Query classification: {classification}")
        
        # Additional debug logging
        print("=== ORIGINAL QUESTION ===", original_question)
        print("=== CLASSIFICATION ===", classification)
        print("=== RETRIEVED CHUNKS ===")
        for i, (score, chunk) in enumerate(retrieval_debug['final_chunks']):
            print(f"Chunk {i+1}:", chunk[:200])
        
        print(f"\n=== EMBEDDINGS ===")
        print(f"Query embedding vector length: {retrieval_debug['embedding_length']}")
        
        print(f"\n=== RETRIEVAL RESULTS ===")
        print(f"Vector search top chunks: {len(retrieval_debug['vector_chunks'])}")
        for i, chunk in enumerate(retrieval_debug['vector_chunks'][:5], 1):
            print(f"\n  Vector chunk {i}:\n{chunk[:200]}...")
        
        print(f"\nBM25 top chunks: {len(retrieval_debug['keyword_chunks'])}")
        for i, (idx, chunk) in enumerate(retrieval_debug['keyword_chunks'][:5], 1):
            print(f"\n  BM25 chunk {i} (index {idx}):\n{chunk[:200]}...")
        
        print(f"\nMerged chunks: {len(retrieval_debug['merged_chunks'])}")
        
        print(f"\n=== FINAL MERGED CHUNKS SENT TO LLM ===")
        print(f"Number of final chunks: {len(retrieval_debug['final_chunks'])}")
        for i, (score, chunk) in enumerate(retrieval_debug['final_chunks'], 1):
            print(f"\nFinal chunk {i} (relevance score: {score}):\n{chunk}\n")

        system_prompt = f"""
You are a precise document assistant.

RULES:
- Answer ONLY from the provided context
- If question is about specific data (grades, scores, names) quote the exact value from context
- If question needs comparing, structure as a clear list
- If answer spans multiple semesters, present in chronological order
- If truly not found say exactly:
"Not found in document" - nothing else
- Never say "based on the context" or "according to the excerpt" - just answer directly
- Keep answers concise, maximum 5 sentences
- For listing questions, use bullet points

Original question: {original_question}
Interpreted question: {rewritten_question}
Query type: {classification}
""".strip()

        print(f"\n=== FULL SYSTEM PROMPT SENT TO LLM ===")
        print(system_prompt)

        history = trim_history(payload.session_id)
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{rewritten_question}"})
        
        # Additional debug logging for final prompt
        final_prompt_text = ""
        for msg in messages:
            final_prompt_text += f"{msg['role']}: {msg['content']}\n\n"
        print("=== FULL PROMPT TO LLM ===", final_prompt_text[:500])

        answer = call_ollama_chat(model=CURRENT_MODEL, messages=messages)
        
        print(f"\n=== LLM RESPONSE ===")
        print(answer)
        print("=" * 80 + "\n")
        
        CHAT_HISTORY.setdefault(payload.session_id, []).extend(
            [
                {"role": "user", "content": original_question},
                {"role": "assistant", "content": answer},
            ]
        )
        trim_history(payload.session_id)
    else:
        print(f"\nDirect mode query: {original_question}")
        rewritten_question = original_question
        context = DOCUMENT_FULL_TEXT[CURRENT_DOC_ID]
        if len(context) > 12000:
            context = context[:12000]

        print(f"\n=== CONTEXT SENT TO LLM ===")
        print(f"Context length: {len(context)} characters")
        print(context[:500] + "..." if len(context) > 500 else context)

        system_prompt = (
            "You have the full document in front of you.\n"
            "Answer directly and precisely.\n"
            "Quote exact values when asked about specific data.\n"
            "Do not say 'based on context' - just answer."
        )
        
        print(f"\n=== SYSTEM PROMPT ===")
        print(system_prompt)

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": f"Document:\n{context}\n\nQuestion: {original_question}"},
        ]
        answer = call_ollama_chat(model=CURRENT_MODEL, messages=messages)
        
        print(f"\n=== LLM RESPONSE ===")
        print(answer)
        print("=" * 80 + "\n")

    return {
        "answer": answer,
        "rewritten_question": rewritten_question,
        "retrieved_context": context,
    }


@app.post("/new-chat")
async def new_chat(payload: NewChatRequest) -> dict[str, str]:
    CHAT_HISTORY[payload.session_id] = []
    return {"status": "cleared"}


@app.get("/current-model")
async def current_model() -> dict[str, str]:
    return {"model": CURRENT_MODEL}


@app.get("/current-mode")
async def current_mode() -> dict[str, str]:
    return {"mode": "rag" if RAG_MODE else "direct"}


@app.get("/debug/prompts")
async def debug_prompts() -> dict[str, Any]:
    return {"enabled": DEBUG_OLLAMA_PROMPTS, "logs": DEBUG_PROMPT_LOGS}


@app.post("/set-model")
async def set_model(payload: ModelRequest) -> dict[str, str]:
    global CURRENT_MODEL, CURRENT_DOC_ID, CURRENT_DOC_TEXT

    if payload.model not in ALLOWED_MODELS:
        raise HTTPException(status_code=400, detail="Unsupported model.")

    CURRENT_MODEL = payload.model
    CURRENT_DOC_ID = None
    CURRENT_DOC_TEXT = ""
    return {"model": CURRENT_MODEL}


@app.post("/set-mode")
async def set_mode(payload: ModeRequest) -> dict[str, str]:
    global RAG_MODE

    normalized_mode = payload.mode.strip().lower()
    if normalized_mode not in {"rag", "direct"}:
        raise HTTPException(status_code=400, detail="Unsupported mode.")

    RAG_MODE = normalized_mode == "rag"
    return {"mode": normalized_mode}


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(UI_DIR / "index.html")


app.mount("/", StaticFiles(directory=UI_DIR, html=True), name="ui")


if __name__ == "__main__":
    try:
        uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
    except KeyboardInterrupt:
        print("\nServer stopped with Ctrl+C.")
