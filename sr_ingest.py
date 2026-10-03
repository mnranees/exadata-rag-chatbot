#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re, shutil
from pathlib import Path
from typing import Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def build_document(data: dict, path: Path) -> Document | None:
    sr_id = clean_text(data.get("sr_id") or data.get("sr_number"))
    title = clean_text(data.get("title") or data.get("subject"))
    subject = clean_text(data.get("subject"))
    description = clean_text(data.get("description"))
    history = clean_text(data.get("text"))
    source = clean_text(data.get("sr_details_url") or data.get("source_url") or str(path))
    source_url = clean_text(data.get("source_url"))

    sections = []
    if sr_id: sections.append(f"SR Number: {sr_id}")
    if title: sections.append(f"Title: {title}")
    if subject and subject != title: sections.append(f"Subject: {subject}")
    if description: sections.append(f"Description:\n{description}")
    if history: sections.append(f"SR History:\n{history}")
    content = "\n\n".join(sections).strip()
    if not content:
        return None

    metadata = {
        "sr_id": sr_id,
        "title": title,
        "subject": subject,
        "source": source,
        "sr_details_url": source,
        "source_url": source_url,
        "captured_at": clean_text(data.get("captured_at")),
        "content_type": "service_request",
        "platform": "Oracle Service Request",
        "source_file": path.name,
    }
    return Document(page_content=content, metadata={k:v for k,v in metadata.items() if v})

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sr-json-dir", type=Path, default=Path("./sr_json"))
    p.add_argument("--chroma-dir", type=Path, default=Path("./vectordb_sr"))
    p.add_argument("--collection", default="service_requests")
    p.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--chunk-size", type=int, default=1200)
    p.add_argument("--chunk-overlap", type=int, default=200)
    args = p.parse_args()

    files = sorted(args.sr_json_dir.glob("*.json"))
    print("JSON files found:", len(files))
    docs, failures = [], []

    for i, path in enumerate(files, 1):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            doc = build_document(data, path)
            if doc is None:
                raise ValueError("No searchable content")
            docs.append(doc)
            if i <= 10 or i % 100 == 0 or i == len(files):
                print(f"[{i}/{len(files)}] {doc.metadata.get('sr_id', path.stem)} - {doc.metadata.get('title','')}")
        except Exception as exc:
            failures.append({"file": path.name, "error": str(exc)})
            print(f"[{i}/{len(files)}] FAILED {path.name}: {exc}")

    print("Loaded SR documents:", len(docs))
    print("Failed files:", len(failures))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print("Chunks:", len(chunks))
    if not chunks:
        raise SystemExit("No chunks created")

    if args.chroma_dir.exists():
        shutil.rmtree(args.chroma_dir)

    embedding = HuggingFaceEmbeddings(
        model_name=args.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    db = Chroma.from_documents(
        chunks,
        embedding=embedding,
        persist_directory=str(args.chroma_dir),
        collection_name=args.collection,
    )

    failure_file = args.chroma_dir.parent / "sr_ingestion_failures.json"
    failure_file.write_text(json.dumps(failures, indent=2), encoding="utf-8")

    print("Collection:", args.collection)
    print("Indexed chunks:", db._collection.count())
    print("Failure log:", failure_file)

    query = "ACFS usage during GI patching on Exadata"
    results = db.similarity_search_with_score(query, k=5)
    print("\nTest query:", query)
    for i, (doc, score) in enumerate(results, 1):
        print(f"--- RESULT {i} score={score} ---")
        print("SR:", doc.metadata.get("sr_id"))
        print("Title:", doc.metadata.get("title"))
        print("Source:", doc.metadata.get("source"))
        print("Text:", doc.page_content[:800])

if __name__ == "__main__":
    main()
