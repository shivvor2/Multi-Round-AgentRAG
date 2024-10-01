# “MultiRound AgentRAG: An Agent-Enhanced Retrieval-Augmented Generation Pipeline for multiround conversation

(This version is usuable but the prompts for the agents are not yet optimized, behavior of the system might not be stable)

We came up with the agentic parts of the project while the non-agentic parts of this work is mostly based on [this paper](https://arxiv.org/pdf/2407.01219) by Fudan university.

For more detailed info, please read the [release blog](https://shivvor2.substack.com/p/building-a-multiround-multihop-rag)

## Usage

A detailed usage example in [main.ipynb](main.ipynb)

Setup the environment and install dependancies in the [setup.ipynb](setup.ipynb)

To use the pipeline, prepare the `PoolOfQueries` object as follows:

```python
pool_of_queries = PoolOfQueries(embedding_function = embedding_function,
                                rerank_function = rerank_function,
                                retrieve_function = retrieve_function,
                                response_function = response_structured_function,
                                verbose = poq_verbose)
```

Then during the conversation:

- Pass in the message to update the context, and invoke it to obtain the current retrieved chunks:

```python
pool_of_queries.update(message_history)
current_context_msg = pool_of_queries.current_context_msg()
```

### Arguements of the `PoolOfQueries` object

Prepare the following functions, if they require other arguments, abstract them away (e.g. using `partial` to supplement some of the arguments)

- embedding_function:
  - Takes in a (list of) text to be embedded(str/ List\[str\])
  - Returns a (list of) tensorlikes (torch.Tensor/ List\[torch.Tensor\] / ndarray).
- reranking_function:
  - Takes in a query (str) and a list of candidates (List[str]) to be ranked
  - Returns a list of reranked results, each in the format of \[result.text, result.score\]
- retrieve_function:
  - Takes in a query (str)
  - Returns a list of related chunks
- response_function:
  - Takes in all chat steps (List[dict], in OAI format), a pydantic datamodel, and (optionally) attribute(s) to retrieve from the datamodel (by default: "response")
  - Returns an instance of the specified attribute determined by your datamodel, or a dictionary containing all the attributes if multiple attributes are specified
    - [Example usage](multiroundRAG/context_management/hyde.py)
- poq_verbose:
  - Boolean to control verbosity of the poq module

## Features

- Not using LLM frameworks (LangChain, llama index), every part should be plug and play
- Retrieved context persistent throughout rounds
- "Sliding Window" retrieval module
- Custom context retrieval/ management mechanism using Agents for multi-round persistence

## To Do List

- Separate the giant notebook into individual files
- Implement source highlighting (equivalently encoding for chunking information)
- Add support different inference providers (local, huggingface etc.)

## Why are there no benchmarks?

Because there are (probably) not Miltihop Miltiround RAG benchmarks. (we can't find any as to date.)  

Most of the pipeline is made with multiround conversation in mind, so if singleround conversation benchmarks are used, the system reduces to the one outlined in the Fudan paper, and will have similar results.
