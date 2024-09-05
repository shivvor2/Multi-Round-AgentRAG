from pymilvus import Collection
from typing import List

# Query
def retrieve(embedded_query, top_k_retrieved, collection: Collection, search_params: dict, padding = [0.5, 0.5]) -> List[str]:
    results = retrieve_vector(embedded_query, top_k_retrieved, collection, search_params)
    results = results[0] # Milvus collection search returns a single item list for some reason
    processed_entities = [process_entity(entity, collection, padding) for entity in results]
    return processed_entities


def retrieve_vector(embedded_query, top_k_retrieved, collection: Collection, search_params: dict):
    results = collection.search(
        data = embedded_query,
        anns_field = "embedding",
        params = search_params,
        limit = top_k_retrieved,
        output_fields = None, # Returns all fields (all implicitly retrievable)
    )
    return results

# process_entity is abstracted out for future modification
def process_entity(entity, collection: Collection, padding = None) -> str: # padding should be an iterator of 2 float e.g. [0.5, 0.5]
    if padding: # Not doing typecheck, hopefully the person invoking the
        lower_bound = entity["chunk_id"] - int(-(-padding[0] // 1)) # int(-(-padding[0] // 1)) Rounds up padding[0]
        upper_bound = entity["chunk_id"] + int(-(-padding[1] // 1)) + 1
        updated_text = []
        oob_lower = False
        oob_upper = False
        for i in range(lower_bound, upper_bound):
            if i == entity["chunk_id"]:
                updated_text.append(entity["chunk_text"])
                continue

            expr = f"doc_id == {entity['doc_id']} && chunk_id == {entity['chunk_id']}"
            results = collection.query(
            expr=expr,
            output_fields=["chunk_text"],
            )

            # Check if have results
            if results:
                updated_text.append(results[0]["chunk_text"])
            else:
                if i == lower_bound:
                    oob_lower = True
                if i == upper_bound - 1:
                    oob_upper = True
                if i != lower_bound and i != upper_bound - 1 and not oob_lower:
                    raise UserWarning(f"Previous Chunks are found but chunk {i} is missing")

        # Truncate edge chunks
        start_fraction = padding[0] - int((padding[0] // 1))
        end_fraction = padding[1] - int((padding[1] // 1))
        if start_fraction != 0 and not oob_lower:
            updated_text[0] = truncate(updated_text[0], start_fraction, False)
        if end_fraction != 0 and not oob_upper:
            updated_text[-1] = truncate(updated_text[-1], start_fraction, True)

        # Join retrieved text
        new_entity = entity
        new_entity["chunk_text"] = " ".join(updated_text)

        return new_entity


# Assumes "Languages with romanic characters", see chunking section
def truncate(text: str, keep_ratio, truncate_end):
    if truncate_end:
        text_truncated = text[:int(len(text)*(1 - keep_ratio))]
        text_truncated = text_truncated.rsplit(" ", 1) # Prevents returning half a word
    else: # Truncates the start
        text_truncated = text[int(len(text)*(1 - keep_ratio)):]
        text_truncated = text_truncated.split(" ", 1)
    return text_truncated