import logging
from typing import List, Callable, Any
from .agent_prompts import *
from response_models import BooleanModel, ListStrModel
from .pair_qualify import qualify_existing_pairs
from .hyde import get_HyDE

class PoolOfQueries():

    def __init__(
        self,
        embedding_function: Callable[[str], Any],
        rerank_function: Callable[[List[Any], str], Any],
        retrieve_function: Callable[[str], Any],
        response_function: Callable[..., Any],
        top_k_retrieve: int = 10,
        top_k_rerank: int = 3,
        max_length_per_split: int = 4,
        chunks: List[dict] = None,
        chunks_cached: List[dict] = None,
        unanswerable: List[dict] = None,
        verbose: bool = False, #Set to 20 if need logging
        **kwargs
    ):
        self.embedder = embedding_function
        self.reranker = rerank_function
        self.retriever = retrieve_function
        self.response_func = response_function
        self.chunks = chunks or []
        self.chunks_cached = chunks_cached or []
        self.unanswerable = unanswerable or []
        self.max_length_per_split = max_length_per_split
        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank = top_k_rerank

        # Setup logging
        self.logger = logging.getLogger(__name__)
        if verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.INFO - 1)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def update(self, messages) -> None:
        self.logger.info("Starting update process")
        if not self._classify_query(messages):
            return

        self._qualify_existing_pairs(messages)
        new_queries = self._generate_new_queries(messages)
        query_hyde = self._generate_hyde(new_queries)
        retrieved_unranked = self._retrieve(new_queries, query_hyde)
        new_pairs = self._rerank(new_queries, retrieved_unranked)
        self._qualify_all_new_generated_pairs(new_pairs)

        self.logger.info("Update process completed")

    def current_context_msg(self):
        msg = format_poq_output(self.chunks, self.unanswerable)
        return msg

    def reset(self):
        self.chunks = []
        self.chunks_cached = []
        self.unanswerable = []

    def _classify_query(self, messages) -> bool:
        self.logger.info("Starting Query Classification")
        msg = [classify_sysprompt] + messages
        response: bool = self.response_func(msg, BooleanModel)
        self.logger.info(f"Query Classification completed: {response}")
        return response

    def _qualify_existing_pairs(self, messages):
        self.logger.info("Starting Query-context pairs qualification")
        relevant_pairs, irrelevant_pairs = qualify_existing_pairs(self.chunks, messages, self.response_func, self.max_length_per_split)
        relevant_pairs_cached, irrelevant_pairs_cached = qualify_existing_pairs(self.chunks_cached, messages, self.response_func, self.max_length_per_split)
        unanswerable_queries, _ = qualify_existing_pairs(self.unanswerable, messages, self.response_func, self.max_length_per_split)

        self.chunks = relevant_pairs + relevant_pairs_cached
        self.chunks_cached = irrelevant_pairs + irrelevant_pairs_cached
        self.unanswerable = unanswerable_queries

        self.logger.info("Query-context pairs qualification completed")

    def _generate_new_queries(self, messages) -> List[str]:
        self.logger.info("Starting generation of new queries")
        msg = [new_query_sysprompt] + messages + [new_query_prompt(self.chunks, self.unanswerable)]
        new_queries: List[str] = self.response_func(msg, ListStrModel)
        self.logger.info(f"Generated {len(new_queries)} new queries")
        return new_queries

    def _generate_hyde(self, new_queries) -> List[dict]:
        self.logger.info("Starting HyDE generation")
        query_hyde: List[dict] = [get_HyDE(x, self.response_func) for x in new_queries]
        self.logger.info("HyDE generation completed")
        return query_hyde

    def _retrieve(self, new_queries, query_hyde) -> List[List[str]]:
        self.logger.info("Starting retrieval process")
        retrieve_queries: List[str] = self._retrieve_queries(new_queries, query_hyde)
        retrieve_embeddings = [self.embedder(query) for query in retrieve_queries]
        retrieved_unranked: List[List[str]] = [self.retriever(embedded_query) for embedded_query in retrieve_embeddings]
        self.logger.info("Retrieval process completed")
        return retrieved_unranked

    def _rerank(self, retrieve_queries, retrieved_unranked) -> List[dict]:
        self.logger.info("Starting reranking process")
        retrieved_ranked = [self.reranker(query, chunks) for query, chunks in zip(retrieve_queries, retrieved_unranked)]
        retrieved_ranked_top_k = [[retrieved_unranked[i][1] for i in ranked_chunks[:self.top_k_rerank]] for ranked_chunks in retrieved_ranked]
        new_pairs = [self._pair_factory(query, context) for query, context in zip(retrieve_queries, retrieved_ranked_top_k)]
        self.logger.info("Reranking process completed")
        return new_pairs

    def _qualify_all_new_generated_pairs(self, new_pairs):
        self.logger.info("Starting qualification of generated pairs")
        new_pair_qualify_bool = [self._qualify_generated_pairs(pair) for pair in new_pairs]
        new_pairs_qualified = [pair for pair, qual in zip(new_pairs, new_pair_qualify_bool) if qual]
        new_unanswerables = [pair["query"] for pair, qual in zip(new_pairs, new_pair_qualify_bool) if not qual]
        self.logger.info("Qualification of generated pairs completed")

        self.chunks.extend(new_pairs_qualified)
        self.unanswerable.extend(new_unanswerables)

    def _qualify_generated_pairs(self, pair):
        msg = [qualify_generated_sysprompt] + [qualify_retrieved_prompt(pair["query"], pair["context"])]
        qual: bool = self.response_func(msg, BooleanModel)
        return qual

    @staticmethod
    def _retrieve_queries(queries, hydes) -> List[str]:
        msg_list = []
        for query, hyde in zip(queries, hydes):
            hyde_response = hyde["response"]
            msg = f"Query: {query}"
            if hyde["generate"]:
                msg = msg + f"Hypothetical Answer: {hyde_response}"
            msg_list.append(msg)
        return msg_list

    @staticmethod
    def _pair_factory(query: str, context: List[str]):
        pair = {
            "query": query,
            "context": context
        }
        return pair