"""
LLM API Embeddings client implementation
"""
import json
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from langchain.embeddings.base import Embeddings
from pydantic import BaseModel, Field  # pylint: disable=no-name-in-module
from requests.adapters import HTTPAdapter


class APIEmbeddings(BaseModel, Embeddings):
    """
    Embeddings model for LLM API client

    Example:
        .. code-block:: python

            emb = APIEmbeddings(
                host_name="your api host name",
                params = {"n_predict": 300, "temp": 0.2}
            )
    """

    host_name: str = "http://localhost:8000"
    request_timeout: Optional[Union[float, Tuple[float, float]]] = 600
    max_retries: int = 3
    params: Dict[str, Any] = Field(default_factory=dict)

    def _embed(self, text: str) -> List[float]:
        """Embed a text using the LLM API.

        Args:
            text: The text to embed.
        Returns:
            Embeddings for the text.
        """
        url = self.host_name + "/embeddings"

        payload = json.dumps({"text": text})
        headers = {"Content-Type": "application/json"}
        with requests.Session() as session:
            session.mount(self.host_name, HTTPAdapter(max_retries=self.max_retries))
            response = session.request(
                "POST", url, headers=headers, data=payload, timeout=self.request_timeout
            )
            return response.json()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using LLM API.
        Args:
            texts: The list of texts to embed.
        Returns:
            List of embeddings, one for each text.
        """
        embeddings = [self._embed(text) for text in texts]
        return [list(map(float, e)) for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using the LLM API.

        Args:
            text: The text to embed.
        Returns:
            Embeddings for the text.
        """

        return self._embed(text)
