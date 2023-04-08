"""Wrappers around LLM API models and embeddings clients."""
from langchain_llm_api.embeddings import APIEmbeddings
from langchain_llm_api.llm import LLMAPI

__all__ = ["LLMAPI", "APIEmbeddings"]
