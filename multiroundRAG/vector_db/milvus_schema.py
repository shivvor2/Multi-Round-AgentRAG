from pymilvus import Collection, FieldSchema, CollectionSchema, DataType

# connections.connect(host='localhost', port='19530')

embedding_dims = 1024
schema_desc = "Document collection with chunking information"

# Kwargs:
# embedding_dim: the dimensions of the embeddings
fields = [
    FieldSchema(name="document_id", dtype= DataType.INT64, is_primary=True),
    FieldSchema(name="chunk_id", dtype= DataType.INT64),
    FieldSchema(name="chunk_length", dtype = DataType.INT64),
    FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="embedding", dtype= DataType.FLOAT_VECTOR, dim = embedding_dims)
]
milvus_schema = CollectionSchema(fields, description=schema_desc)

# collection = Collection(name="documents", schema=schema)