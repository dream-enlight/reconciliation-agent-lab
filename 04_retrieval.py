"""Inspect local document indexing and retrieval before connecting an LLM.

python3 04_retrieval.py build
python3 04_retrieval.py search "Why can a booked exposure be missing from System B?"
"""

import argparse
import hashlib
import re
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "retrieval_db"
COLLECTION_NAME = "reconciliation_procedures_minilm"


# PART 1: Read ONLY procedures. Keep each short rule as a complete chunk.
def load_chunks():
    chunks = []
    for path in sorted((ROOT / "procedures").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        header = text.split("\n## ", 1)[0]
        fields = dict(re.findall(r"^(procedure_id|version|effective_date|status):\s*(.+)$", header, re.M))
        if fields.get("status") != "active":
            continue
        title = text.splitlines()[0].removeprefix("# ")
        sections = re.split(r"^## ", text, flags=re.M)[1:]
        for section in sections:
            heading, body = section.split("\n", 1)
            section_id = heading.split(":", 1)[0].strip()
            chunks.append({
                "id": f"{section_id}:v{fields['version']}",
                "text": f"{title} | version {fields['version']}\n{heading}\n{body.strip()}",
                "metadata": {
                    "section_id": section_id,
                    "procedure_id": fields["procedure_id"],
                    "version": int(fields["version"]),
                    "effective_date": fields["effective_date"],
                    "status": fields["status"],
                    "source": path.name,
                },
            })
    if not chunks:
        raise SystemExit("No active procedure sections found in procedures/.")
    if len({chunk["id"] for chunk in chunks}) != len(chunks):
        raise SystemExit("Duplicate active section/version IDs. Check procedures/.")
    return chunks


def source_digest(chunks):
    # Detect edits so a search cannot silently use an outdated index.
    import json
    return hashlib.sha256(json.dumps(chunks, sort_keys=True).encode()).hexdigest()


# PART 2: Embedding happens HERE, before we give the vectors to the database.
def build_index():
    chunks = load_chunks()
    for chunk in chunks:
        print(f"CHUNK: {chunk['id']} | {chunk['metadata']['source']}")
    print("\nEmbedding model: all-MiniLM-L6-v2, running locally.", flush=True)
    print("First run downloads the model files; later runs reuse them.", flush=True)
    embed = DefaultEmbeddingFunction()
    vectors = embed([chunk["text"] for chunk in chunks])
    print("Vector dimensions:", len(vectors[0]))
    print("First vector, first six values:", [round(float(v), 4) for v in vectors[0][:6]])

    # PART 3: Store each vector together with its original text and metadata.
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,  # We explicitly supply all embeddings.
        configuration={"hnsw": {"space": "cosine"}},
    )
    ids = [chunk["id"] for chunk in chunks]
    collection.upsert(
        ids=ids,
        embeddings=vectors,
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
    )
    old_ids = collection.get(include=[])["ids"]
    stale_ids = [item for item in old_ids if item not in set(ids)]
    if stale_ids:
        collection.delete(ids=stale_ids)
    collection.modify(metadata={"source_digest": source_digest(chunks)})
    print(f"\nStored {collection.count()} chunks in: {DB_PATH}")
    print("Build complete. No chat model was called.")


def search_procedures(query, top_k=2):
    if not query.strip():
        raise ValueError("Provide a non-empty query.")
    client = chromadb.PersistentClient(path=str(DB_PATH))
    try:
        collection = client.get_collection(COLLECTION_NAME, embedding_function=None)
    except chromadb.errors.NotFoundError:
        raise SystemExit("Build the index first: python3 04_retrieval.py build")
    if (collection.metadata or {}).get("source_digest") != source_digest(load_chunks()):
        raise SystemExit("Procedures changed. Rebuild: python3 04_retrieval.py build")

    # PART 4: Embed the query with the SAME model, then search nearby vectors.
    embed = DefaultEmbeddingFunction()
    query_vectors = embed([query])
    found = collection.query(
        query_embeddings=query_vectors,
        n_results=min(top_k, collection.count()),
        where={"status": "active"},
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"id": item_id, "text": text, "metadata": metadata, "distance": float(distance)}
        for item_id, text, metadata, distance in zip(
            found["ids"][0], found["documents"][0],
            found["metadatas"][0], found["distances"][0],
        )
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["build", "search"])
    parser.add_argument("query", nargs="?")
    args = parser.parse_args()
    if args.action == "build":
        build_index()
        return
    if not args.query:
        parser.error('search needs a quoted query, e.g. search "Why is an exposure missing?"')
    print("QUERY:", args.query, flush=True)
    for rank, hit in enumerate(search_procedures(args.query), start=1):
        print(f"\nRESULT {rank}: {hit['id']}")
        print("Source:", hit["metadata"]["source"])
        print(f"Cosine distance: {hit['distance']:.4f} (lower = closer; not a confidence score)")
        print(hit["text"])
    print("\nRetrieved passages only. No answer has been generated by a chat model.")


if __name__ == "__main__":
    main()
