# Imports
from sentence_transformers import SentenceTransformer
from typing import Union, List

def get_embeddings(texts: Union[str, List[str]],
                   embedding_model: SentenceTransformer,
                   **kwargs):
    return embedding_model.encode(texts, **kwargs)

# # Torch Implementation
# import torch
# from sklearn.preprocessing import normalize

# # Select embedding model
# embedding_model_name = "dunzhang/stella_en_1.5B_v5"

# embedding_model = SentenceTransformer(embedding_model_name, trust_remote_code=True).cuda()

# # embedding_model.encode()
# def get_embeddings(queries, model, tokenizer, vector_linear):
#     with torch.no_grad():
#         input_data = tokenizer(queries, padding="longest", truncation=True, max_length=512, return_tensors="pt")
#         input_data = {k: v.cuda() for k, v in input_data.items()}
#         attention_mask = input_data["attention_mask"]
#         last_hidden_state = model(**input_data)[0]
#         last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
#         query_vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
#         query_vectors = normalize(vector_linear(query_vectors).cpu().numpy())
#     return query_vectors

# Usage:
# model_dir = "stella_en_1.5B_v5"
# vector_dim = 1024
# vector_linear_directory = f"2_Dense_{vector_dim}"
# embedding_model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).cuda().eval()
# embedding_tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
# vector_linear = torch.nn.Linear(in_features=embedding_model.config.hidden_size, out_features=vector_dim)
# vector_linear_dict = {
#     k.replace("linear.", ""): v for k, v in
#     torch.load(os.path.join(model_dir, f"{vector_linear_directory}/pytorch_model.bin")).items()
# }
# vector_linear.load_state_dict(vector_linear_dict)
# vector_linear.cuda()

# embedding_function = partial(get_embeddings, model = embedding_model, tokenizer = embedding_tokenizer, vector_linear = vector_linear)