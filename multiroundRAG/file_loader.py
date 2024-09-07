import os
from unstructured.partition.auto import partition
from pymilvus import Collection
from .chunking import sentence_level_chunking
from typing import Callable, List, Union
from torch import Tensor
from numpy import ndarray
from xxhash import xxh128_intdigest

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

# Example usage for doc_ig_gen_func
# import xxhash
# <Some Code>
# store_and_embed_documents(documents, collection, doc_id_gen_func = xxhash.xxh128_hexdigest)

def store_and_embed_documents(documents: list, collection: Collection, embedding_func: Callable, chunking_func: Callable = sentence_level_chunking, doc_id_gen_func: Union[Callable, None] = xxh128_intdigest, batch_encoding: bool = False):
    for i, doc in enumerate(documents):
        doc_id = doc_id_gen_func(doc) if doc_id_gen_func else i
        index_document(doc, collection, embedding_func, chunking_func, doc_id, batch_encoding)

# use partial to create an embedding function "embedder" that eats 1 arguement only and returns an embedding (str -> torch.tensor (or other equivalent class))
def index_document(document, collection: Collection, embedding_func: Callable, chunking_func: Callable, doc_id: int, batch_encoding: bool = False):
    data = []
    chunks = chunking_func(document) # returns list of dicts with fields: "text" and "chunk_length"
    embeddings_all = embedding_func(chunks) if batch_encoding else None
    for i, chunk in enumerate(chunks): # Should we consider making this it's own function?
        embedding = embeddings_all[i] if embeddings_all else embedding_func(chunk["text"])
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