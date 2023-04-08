"""
LLM API model client implementation
"""
import json
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import requests
from langchain.llms.base import LLM
from pydantic import BaseModel, Field  # pylint: disable=no-name-in-module
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException
from sseclient import SSEClient


class LLMAPI(LLM, BaseModel):
    """A wrapper for LLM API client.

    Example:
        .. code-block:: python

            llm = LLMAPI(
                host_name="http://localhost:8000",
                params = {"n_predict": 300, "temp": 0.2}
            )
    """

    streaming: bool = False
    host_name: str = "http://localhost:8000"
    request_timeout: Optional[Union[float, Tuple[float, float]]] = 600
    max_retries: int = 3
    params: Dict[str, Any] = Field(default_factory=dict)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call the LLM API model and return the output.

        Args:
            prompt: The prompt to use for generation.
            stop: A list of strings to stop generation when encountered.
        Returns:
            The generated text.
        Example:
            .. code-block:: python
                llm = LLMAPI(
                    host_name="http://localhost:8000",
                    params = {"n_predict": 300, "temp": 0.2}
                )
                llm("This is a prompt.")
        """

        self.params["stop"] = stop or []

        payload = json.dumps({"prompt": prompt, "params": self.params})
        headers = {"Content-Type": "application/json"}

        if self.streaming:
            url = self.host_name + "/agenerate"
            headers["Accept"] = "text/event-stream"
            with requests.Session() as session:
                session.mount(
                    self.host_name, HTTPAdapter(max_retries=self.max_retries)
                )
                with session.post(
                    url,
                    stream=True,
                    headers=headers,
                    data=payload,
                    timeout=self.request_timeout,
                ) as response:
                    try:
                        client = SSEClient(response)
                        current_completion = ""
                        for event in client.events():
                            current_completion += event.data
                            self.callback_manager.on_llm_new_token(
                                token=event.data, verbose=self.verbose
                            )
                        return current_completion
                    except RequestException as exp:
                        raise RuntimeError() from exp
                    finally:
                        response.close()
                        client.close()

        url = self.host_name + "/generate"
        with requests.Session() as session:
            session.mount(
                self.host_name, HTTPAdapter(max_retries=self.max_retries)
            )
            response = session.request(
                "POST",
                url,
                headers=headers,
                data=payload,
                timeout=self.request_timeout,
            )
            return response.text

    async def _acall(
        self, prompt: str, stop: Optional[List[str]] = None
    ) -> str:
        """Call to LLM API endpoint asynchronously.

        Args:
            prompt: The prompt to use for generation.
            stop: A list of strings to stop generation when encountered.
        Returns:
            The generated text.
        Example:
            .. code-block:: python
                llm = LLMAPI(
                    host_name="http://localhost:8000",
                    params = {"n_predict": 300, "temp": 0.2},
                    stream=True,
                    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
                )
                llm("This is a prompt.")
        """
        url = self.host_name + "/agenerate"

        self.params["stop"] = stop or []

        payload = json.dumps({"prompt": prompt, "params": self.params})
        headers = {"Content-Type": "application/json"}

        if self.streaming:
            url = self.host_name + "/agenerate"
            headers["Accept"] = "text/event-stream"
            with requests.Session() as session:
                session.mount(
                    self.host_name, HTTPAdapter(max_retries=self.max_retries)
                )
                with session.post(
                    url,
                    stream=True,
                    headers=headers,
                    data=payload,
                    timeout=self.request_timeout,
                ) as response:
                    try:
                        client = SSEClient(response)
                        current_completion = ""
                        async for event in client.events():
                            current_completion += event.data
                            if self.callback_manager.is_async:
                                await self.callback_manager.on_llm_new_token(
                                    token=event.data, verbose=self.verbose
                                )
                            else:
                                self.callback_manager.on_llm_new_token(
                                    token=event.data, verbose=self.verbose
                                )
                        return current_completion
                    except RequestException as exp:
                        raise RuntimeError() from exp
                    finally:
                        response.close()
                        client.close()

        with requests.Session() as session:
            session.mount(
                self.host_name, HTTPAdapter(max_retries=self.max_retries)
            )
            response = session.request(
                "POST",
                url,
                headers=headers,
                data=payload,
                timeout=self.request_timeout,
            )
            return response.text

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"host_name": self.host_name}, **self.params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "llm-api"
