# “MultiRound AgentRAG: An Agent-Enhanced Retrieval-Augmented Generation Pipeline for multiround conversation

(This version is usuable but the prompts for the agents are not yet optimized, behavior of the system might not be stable)

We came up with the agentic parts of the project while the non-agentic parts of this work is mostly based on [this paper](https://arxiv.org/pdf/2407.01219) by Fudan university. 

## Features

- Not using LLM frameworks (LangChain, llama index), every part should be plug and play
- Retrieved context persistent throughout rounds
- "Sliding Window" retrieval module
- Custom context retrieval/ management mechanism using Agents for multi-round persistence

## To Do List

- Separate the giant notebook into individual files
- Implement source highlighting (equivalently encoding for chunking information)
- Add support different inference providers (local, huggingface etc.)

## Benchmarks?

The implementation of this RAG system is designed for multi-round conversation in mind.

However, as to date, no Miltihop Miltiround RAG benchmarks exists

Once such a benchmark gets published, the system will be benchmarked against it.
