import os
from unstructured.partition.auto import partition
from pymilvus import Collection
from .chunking import sentence_level_chunking
from typing import Callable

def read_file(file_path):
    # Read a file using Unstructured library.
    elements = partition(filename=file_path)
    # Process elements as needed, e.g., extract text
    return ' '.join([el.text for el in elements])

def read_directory(directory_path, recursive=True):
    # Recursively traverse directory and read files.
    documents = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                content = read_file(file_path)
                documents.append(content)
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")

        if not recursive:
            break  # Don't process subdirectories if recursive is False

    return documents

# Can customize doc_id later
def store_and_embed_documents(documents: list, collection: Collection, embedding_func: Callable, chunker_kwargs: dict = dict()):
    for i, doc in enumerate(documents):
        index_document(doc, collection, embedding_func, i, chunker_kwargs)

# use partial to create an embedding function "embedder" that eats 1 arguement only and returns an embedding (str -> torch.tensor (or other equivalent class))
def index_document(document, collection: Collection, embedding_func: Callable, doc_id: int, chunker_kwargs: dict = dict()):
    data = []
    chunks = sentence_level_chunking(document, **chunker_kwargs) # returns list of dicts with fields: "text" and "chunk_length"
    for i, chunk in enumerate(chunks): # Should we consider making this it's own function?
        embedding = embedding_func(chunk["text"])
        entity_dict = {
            "document_id": doc_id,
            "chunk_id": i,
            "chunk_length": int(chunk["chunk_length"]),
            "chunk_text": chunk["text"],
            "embedding": list(embedding),
        }
        data.append(entity_dict)
    collection.insert(data)
    # collection.flush()  # might need to flush in production

