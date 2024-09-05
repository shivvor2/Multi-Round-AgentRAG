from pygaggle.rerank.base import Reranker, Query, Text
from pygaggle.rerank.transformer import MonoT5
from typing import List, Tuple

# Define a function to rerank the results

def rerank_results(query: str, candidates: List[str], top_k: int, reranker: Reranker) -> List[Tuple[str, int]]:
  correct_query = Query(query)
  correct_candidates = [Text(candidate) for candidate in candidates]
  reranked_results = reranker.rescore(correct_query, correct_candidates)
  reranked_results = [[result.text, result.score] for result in reranked_results]
  # reranked_results.sort(key=lambda x: -x[1])
  reranked_results = reranked_results[:top_k]
  return reranked_results