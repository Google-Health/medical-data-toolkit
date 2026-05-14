# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""LLM Client interface and implementations for Gemini and MedGemma."""

import abc
import base64
import json
import logging
from typing import Any, Dict, List, Optional, Union

from google import genai
import google.auth
import google.auth.transport.requests
from google.genai import types
import litellm
from litellm import get_supported_openai_params
import pydantic
import requests
import tenacity


Request = google.auth.transport.requests.Request


class ResponseParsingError(Exception):
  """Raised when LLM response cannot be parsed or validated."""

  pass


class LLMClient(abc.ABC):
  """Abstract base class for LLM clients."""

  def __init__(self, verbose: bool = False):
    self.verbose = verbose

  @property
  @abc.abstractmethod
  def supports_pdf(self) -> bool:
    """Returns True if the LLM client supports PDF input."""
    pass

  @abc.abstractmethod
  def generate_content(
      self,
      contents: List[Union[str, Any]],
      schema: Optional[Any] = None,
      config: Optional[Dict[str, Any]] = None,
  ) -> Any:
    """Generates content using the specified configuration."""


def create_llm_client(
    client_type: str, model_name: str, api_key: str
) -> LLMClient:
  """Creates an LLM client based on the client type."""
  if client_type == "gemini":
    return GeminiClient(api_key=api_key, model=model_name)
  elif client_type == "gemma":
    return GemmaClient(api_key=api_key, model=model_name)
  elif client_type == "litellm":
    return LiteLLMClient(model=model_name, api_key=api_key)
  else:
    raise ValueError(f"Unsupported client type: {client_type}")


class GeminiClient(LLMClient):
  """Client for Gemini API via the genai library."""

  def __init__(
      self,
      api_key: str,
      model: str,
      config: Optional[Dict[str, Any]] = None,
      verbose: bool = False,
  ):
    super().__init__(verbose=verbose)
    self.client = genai.Client(api_key=api_key)
    self.model = model
    self.config = config or {}
    self.config["response_mime_type"] = "application/json"

  @property
  def supports_pdf(self) -> bool:
    return True

  def generate_content(
      self,
      contents: List[Union[str, Any]],
      schema: Optional[Any] = None,
      config: Optional[Dict[str, Any]] = None,
  ) -> Any:
    """Calls the Gemini API's generate_content method."""
    merged_config = self.config.copy()
    if config:
      merged_config.update(config)

    if schema:
      merged_config["response_schema"] = schema

    return self.client.models.generate_content(
        model=self.model, contents=contents, config=merged_config
    )


class GemmaClient(LLMClient):
  """Client for Gemma API via the genai library."""

  def __init__(
      self,
      api_key: str,
      model: str,
      config: Optional[Dict[str, Any]] = None,
      verbose: bool = False,
  ):
    super().__init__(verbose=verbose)
    self.client = genai.Client(api_key=api_key)
    self.model = model
    self.config = config or {}
    self.config["response_mime_type"] = "application/json"

  @property
  def supports_pdf(self) -> bool:
    return False

  def generate_content(
      self,
      contents: List[Union[str, Any]],
      schema: Optional[Any] = None,
      config: Optional[Dict[str, Any]] = None,
  ) -> Any:
    """Calls the Gemma API's generate_content method."""
    merged_config = self.config.copy()
    if config:
      merged_config.update(config)

    if schema:
      merged_config["response_schema"] = schema

    return self.client.models.generate_content(
        model=self.model, contents=contents, config=merged_config
    )


class MockResponse:
  """A mock response class to mimic genai response structure."""

  def __init__(self, text, parsed=None):
    self.text = text
    self.parsed = parsed


def _parse_structured_response(text_response: str, response_schema: Any) -> Any:
  """Parses structured JSON from LLM response text."""
  clean_text = text_response
  parsed_json = None
  try:
    clean_text = text_response.strip()
    unused95_tag = "<unused95>"
    if unused95_tag in clean_text:
      clean_text = clean_text[
          clean_text.find(unused95_tag) + len(unused95_tag) :
      ].strip()

    ctrl95_tag = "<ctrl95>"
    if ctrl95_tag in clean_text:
      clean_text = clean_text[
          clean_text.find(ctrl95_tag) + len(ctrl95_tag) :
      ].strip()

    # Robust JSON extraction: look for ```json and the following ```
    json_start_marker = "```json"
    json_end_marker = "```"

    start_idx = clean_text.find(json_start_marker)
    if start_idx != -1:
      start_idx += len(json_start_marker)
      end_idx = clean_text.find(json_end_marker, start_idx)
      if end_idx != -1:
        clean_text = clean_text[start_idx:end_idx].strip()
    elif clean_text.startswith("```"):
      # Handle case where it's just ``` without 'json'
      start_idx = 3
      end_idx = clean_text.find(json_end_marker, start_idx)
      if end_idx != -1:
        clean_text = clean_text[start_idx:end_idx].strip()

    # Always parse as JSON first to allow for flexible unwrapping.
    if not clean_text.startswith("{") and not clean_text.startswith("["):
      raise ResponseParsingError(
          "Cleaned response does not start with JSON object or array. Raw"
          f" response: {text_response}"
      )
    parsed_json = json.loads(clean_text)

    # Handle cases where LLM wraps response in an outer key matching
    # schema name.
    if (
        isinstance(parsed_json, dict)
        and len(parsed_json) == 1
        and hasattr(response_schema, "__name__")
    ):
      key = list(parsed_json.keys())[0]
      schema_name = response_schema.__name__.lower()
      normalized_key = key.lower().replace("_", "").replace("-", "")

      # Match if key is schema name, or generic "results" / "data"
      if normalized_key == schema_name.replace("_", "") or normalized_key in [
          "results",
          "response",
          "data",
          "output",
      ]:
        if isinstance(parsed_json[key], (dict, list)):
          logging.info("Unwrapping response from outer key: %s", key)
          parsed_json = parsed_json[key]

    # Handle Pydantic validation and optional list unwrapping together.
    if isinstance(response_schema, type) and issubclass(
        response_schema, pydantic.BaseModel
    ):
      if isinstance(parsed_json, list):
        if len(parsed_json) == 1:
          logging.info("Unwrapping response from list of length 1")
          parsed_json = parsed_json[0]
        else:
          raise ResponseParsingError(
              "Expected a single JSON object but received a list of size"
              f" {len(parsed_json)}."
          )

      # Validate `parsed_json` against `response_schema`.
      # A `ValidationError` is caught and re-raised as `ResponseParsingError`
      # for consistent error handling by clients.
      return response_schema.model_validate(parsed_json)
    return parsed_json

  except (
      pydantic.ValidationError,
      json.JSONDecodeError,
      TypeError,
      AttributeError,
  ) as e:
    err_msg = "JSON decoding or processing failed"
    if isinstance(e, pydantic.ValidationError):
      err_msg = "Pydantic validation failed"
      logging.error(
          "%s. Errors:\n%s", err_msg, json.dumps(e.errors(), indent=2)
      )
      logging.error(
          "JSON content that failed validation:\n%s",
          json.dumps(parsed_json, indent=2),
      )
    else:
      logging.error("%s: %s", err_msg, e)

    logging.exception(
        "Failed to parse response. Raw response text: %s", text_response
    )
    raise ResponseParsingError(f"Response failed: {e}") from e


class LiteLLMClient(LLMClient):
  """Client for multiple models using LiteLLM."""

  def __init__(
      self,
      model: str,
      api_base: Optional[str] = None,
      api_key: Optional[str] = None,
      temperature: float = 0.0,
      config: Optional[Dict[str, Any]] = None,
      verbose: bool = False,
      timeout: float = 300.0,
      max_retries: int = 3,
      supports_pdf: bool = False,
  ):
    super().__init__(verbose=verbose)
    self.model = model
    self.api_base = api_base
    self.api_key = api_key
    self.temperature = temperature
    self.config = config or {}
    self.timeout = timeout
    self.max_retries = max_retries
    self._supports_pdf = supports_pdf

  @property
  def supports_pdf(self) -> bool:
    return self._supports_pdf

  def generate_content(
      self,
      contents: List[Union[str, Any]],
      schema: Optional[Any] = None,
      config: Optional[Dict[str, Any]] = None,
  ) -> Any:
    """Calls LiteLLM completion API."""
    merged_config = self.config.copy()
    if config:
      merged_config.update(config)

    local_contents = contents.copy()

    supported_params = get_supported_openai_params(model=self.model) or []

    # Add schema to the prompt if the client does not support response_format.
    if schema and "response_format" not in supported_params:
      if hasattr(schema, "model_json_schema"):
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        schema_prompt = (
            "\n\nIMPORTANT: The response must be a valid JSON object matching"
            f" the following JSON schema:\n{schema_json}"
        )
        local_contents.append(schema_prompt)

    messages = []
    user_content = []
    for part in local_contents:
      if isinstance(part, str):
        user_content.append({"type": "text", "text": part})
      elif isinstance(part, types.Part):
        if part.text:
          user_content.append({"type": "text", "text": part.text})
        elif part.inline_data:
          data = part.inline_data.data
          base64_data = base64.b64encode(data).decode("utf-8")
          user_content.append({
              "type": "image_url",
              "image_url": {
                  "url": (
                      f"data:{part.inline_data.mime_type};base64,{base64_data}"
                  )
              },
          })
      elif isinstance(part, dict) and "role" in part:
        if user_content:
          messages.append({"role": "user", "content": user_content})
          user_content = []
        messages.append(part)

    if user_content:
      messages.append({"role": "user", "content": user_content})

    completion_args = {
        "model": self.model,
        "messages": messages,
        "temperature": self.temperature,
        "timeout": self.timeout,
    }

    if self.api_base:
      completion_args["api_base"] = self.api_base
    if self.api_key:
      completion_args["api_key"] = self.api_key
    if schema and "response_format" in supported_params:
      completion_args["response_format"] = schema

    if self.verbose:
      logging.info("LiteLLM Request: model=%s", self.model)

    def _make_request():
      response = litellm.completion(**completion_args)
      text_response = response.choices[0].message.content

      parsed_obj = None
      if schema:
        if (
            hasattr(response.choices[0].message, "parsed")
            and response.choices[0].message.parsed
        ):
          parsed_obj = response.choices[0].message.parsed
        else:
          parsed_obj = _parse_structured_response(text_response, schema)

      return MockResponse(text=text_response, parsed=parsed_obj)

    def _custom_wait(retry_state):
      exc = retry_state.outcome.exception()
      if isinstance(exc, litellm.RateLimitError):
        return tenacity.wait_random_exponential(multiplier=1, max=60)(
            retry_state
        )
      elif isinstance(exc, ResponseParsingError):
        return 1
      return 0

    def _before_sleep(retry_state):
      exc = retry_state.outcome.exception()
      wait_time = _custom_wait(retry_state)
      if isinstance(exc, litellm.RateLimitError):
        logging.warning(
            "Rate limit hit. Retrying in %ds... (Attempt %d/%d)",
            wait_time,
            retry_state.attempt_number,
            self.max_retries,
        )
      elif isinstance(exc, ResponseParsingError):
        logging.warning(
            "Response parsing failed. Retrying in 1s... (Attempt %d/%d)",
            retry_state.attempt_number,
            self.max_retries,
        )

    retryer = tenacity.Retrying(
        stop=tenacity.stop_after_attempt(self.max_retries),
        wait=_custom_wait,
        retry=tenacity.retry_if_exception_type(
            (litellm.RateLimitError, ResponseParsingError)
        ),
        before_sleep=_before_sleep,
        reraise=True,
    )

    try:
      return retryer(_make_request)
    except Exception as e:
      if not isinstance(e, (litellm.RateLimitError, ResponseParsingError)):
        logging.exception("LiteLLM request failed: %s", e)
      raise e
