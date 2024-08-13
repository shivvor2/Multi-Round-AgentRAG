# Multi-Round RAG (framework less) (WIP)

(This is a WIP, a usable version should be coming this month)

This work is based on [this paper](https://arxiv.org/pdf/2407.01219) by Fudan university, we may add more

## Features

- Not using LLM frameworks (LangChain, llama index), every part should be plug and play
- Retrieved context persistent throughout rounds
- "Sliding Window" retrieval module
- Custom retrieval mechanism for multi-round persistence

## To Do List

- Finish the main workflow
- Separate the giant notebook into individual files
- Implement source highlighting (equivalently encoding for chunking information)
- Add support different inference providers (local, huggingface etc.)
- improve the UI idk

## Benchmarks?

The implementation of this RAG system is designed for multi-round conversation in mind.
