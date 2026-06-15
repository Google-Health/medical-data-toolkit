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
"""LLM Client interface and implementations for Gemini and any LiteLLM supported models."""

import abc
import base64
import contextvars
import copy
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Union
from google import genai
import google.auth
import google.auth.transport.requests
from google.genai import types
import litellm
from litellm import get_supported_openai_params
import pydantic
import requests
import tenacity

token_usage_var: contextvars.ContextVar[
    Optional[List["LLMUsage"]]
] = contextvars.ContextVar("token_usage", default=None)


Request = google.auth.transport.requests.Request
_THOUGHT_PATTERN = re.compile(
    r"<\|channel>thought\n(.*?)\n<channel\|>", re.DOTALL
)


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
      post_process: Optional[Callable[[Any], Any]] = None,
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


class GenAIClientBase(LLMClient):
  """Base client for GenAI API models."""

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

  def generate_content(
      self,
      contents: List[Union[str, Any]],
      schema: Optional[Any] = None,
      config: Optional[Dict[str, Any]] = None,
      post_process: Optional[Callable[[Any], Any]] = None,
  ) -> Any:
    """Calls the GenAI API's generate_content method.

    Args:
      contents: A list of message parts, which can be strings or `types.Part`
        objects.
      schema: An optional Pydantic model or dict representing the expected JSON
        schema for the response.
      config: An optional dictionary of configuration parameters to override or
        add to the client's default configuration.
      post_process: An optional callable that takes the parsed JSON object as
        input and returns a processed JSON object. This is called if Pydantic
        validation fails.

    Returns:
      The response object from the `genai.Client.models.generate_content` call.
    """
    merged_config = self.config.copy()
    if config:
      merged_config.update(config)

    if schema:
      merged_config["response_schema"] = schema

    raw_response = self.client.models.generate_content(
        model=self.model, contents=contents, config=merged_config
    )

    usage = None
    if hasattr(raw_response, "usage_metadata") and raw_response.usage_metadata:
      usage = LLMUsage(
          prompt_tokens=raw_response.usage_metadata.prompt_token_count,
          completion_tokens=raw_response.usage_metadata.candidates_token_count,
          total_tokens=raw_response.usage_metadata.total_token_count,
      )

    if usage:
      usage_list = token_usage_var.get()
      if usage_list is not None:
        usage_list.append(usage)

    return MockResponse(
        text=raw_response.text,
        parsed=raw_response.parsed,
        usage=usage,
    )


class GeminiClient(GenAIClientBase):
  """Client for Gemini API via the genai library."""

  @property
  def supports_pdf(self) -> bool:
    return True


class GemmaClient(GenAIClientBase):
  """Client for Gemma API via the genai library."""

  @property
  def supports_pdf(self) -> bool:
    return False


class LLMUsage(pydantic.BaseModel):
  prompt_tokens: int
  completion_tokens: int
  total_tokens: int


class MockResponse:
  """A response class to mimic genai response structure and hold metadata."""

  def __init__(
      self,
      text: str,
      parsed: Any = None,
      thinking: Optional[str] = None,
      usage: Optional[LLMUsage] = None,
  ):
    """Initializes the MockResponse.

    Args:
      text: The raw text response from the model.
      parsed: The parsed object from the response, often a Pydantic model.
      thinking: Optional, the extracted thinking process if available.
      usage: Optional, the token usage metadata.
    """
    self.text = text
    self.parsed = parsed
    self.thinking = thinking
    self.usage = usage


def _clean_and_extract_json_string(text_response: str) -> str:
  """Cleans LLM response text and extracts JSON string."""
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
  return clean_text


def _unwrap_json(parsed_json: Any, response_schema: Any) -> Any:
  """Unwraps JSON if it's nested in a single key like {'response': ...}."""
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
        logging.warning("Unwrapping response from outer key: %s", key)
        return parsed_json[key]
  return parsed_json


def _validate_and_process_json(
    parsed_json: Any,
    response_schema: Any,
    post_process: Optional[Callable[[Any], Any]] = None,
) -> Any:
  """Validates parsed_json against response_schema, applying post_process on failure."""
  if isinstance(response_schema, type) and issubclass(
      response_schema, pydantic.BaseModel
  ):
    try:
      return response_schema.model_validate(parsed_json)
    except pydantic.ValidationError as e:
      if post_process:
        logging.warning("Validation failed. Applying post_process.")
        processed_json = post_process(parsed_json)
        return response_schema.model_validate(processed_json)
      raise e
  return parsed_json


def parse_structured_response(
    text_response: str,
    response_schema: Any,
    post_process: Optional[Callable[[Any], Any]] = None,
) -> Any:
  """Parses structured JSON from LLM response text."""
  parsed_json = None
  try:
    clean_text = _clean_and_extract_json_string(text_response)

    if not clean_text.startswith("{") and not clean_text.startswith("["):
      snippet = (
          text_response[:500] + "..."
          if len(text_response) > 500
          else text_response
      )
      raise ResponseParsingError(
          "Cleaned response does not start with JSON object or array. Raw"
          f" response snippet: {snippet}"
      )

    parsed_json = json.loads(clean_text)
    unwrapped_json = _unwrap_json(parsed_json, response_schema)
    return _validate_and_process_json(
        unwrapped_json, response_schema, post_process
    )

  except (
      pydantic.ValidationError,
      json.JSONDecodeError,
      TypeError,
      AttributeError,
  ) as e:
    err_msg = "JSON decoding or processing failed"
    if isinstance(e, pydantic.ValidationError):
      err_msg = "Pydantic validation failed"
      logging.error("%s", err_msg)
      logging.error("Errors:\n%s", json.dumps(e.errors(), indent=2))
      logging.debug(
          "JSON content that failed validation:\n%s",
          json.dumps(parsed_json, indent=2),
      )
    elif isinstance(e, json.JSONDecodeError):
      err_msg = "JSON decoding failed"
      logging.error("%s: %s", err_msg, e)
      context_size = 100
      start = max(0, e.pos - context_size)
      end = min(len(e.doc), e.pos + context_size)
      snippet = e.doc[start:end]
      logging.error(
          "JSON decoding error context around position %d:\n...%s...",
          e.pos,
          snippet,
      )
    else:
      logging.error("%s: %s", err_msg, e)
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
      enable_thinking: bool = False,
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
    self.enable_thinking = enable_thinking

  @property
  def supports_pdf(self) -> bool:
    return self._supports_pdf

  def _prepare_messages(
      self,
      contents: List[Union[str, Any]],
      schema: Optional[Any],
      merged_config: Dict[str, Any],
      thinking_enabled: bool,
  ) -> List[Dict[str, Any]]:
    """Prepares messages in the LiteLLM/OpenAI format from contents.

    Args:
      contents: A list of message parts, can be strings, types.Part, or dicts
        with a "role".
      schema: An optional Pydantic schema to inject into the prompt.
      merged_config: The merged configuration dictionary.
      thinking_enabled: Whether to inject a thinking token in the system
        message.

    Returns:
      A list of dictionaries, where each dictionary represents a message in the
      LiteLLM/OpenAI format.
    """
    local_contents = contents.copy()

    # Always inject the schema to the prompt to provide semantic context.
    if schema:
      if hasattr(schema, "model_json_schema"):
        schema_json = json.dumps(schema.model_json_schema())
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

    # Handle system_instruction from config
    system_instruction_str = merged_config.get("system_instruction", "")
    if system_instruction_str:
      messages.insert(0, {"role": "system", "content": system_instruction_str})

    if thinking_enabled:
      # Inject thinking token into system message
      system_msg = None
      for m in messages:
        if m.get("role") == "system":
          system_msg = m
          break

      if system_msg:
        # Assumes content in system_msg is a string.
        if system_msg["content"]:
          system_msg["content"] = f"<|think|> {system_msg['content']}"
        else:
          system_msg["content"] = "<|think|>"
      else:
        messages.insert(0, {"role": "system", "content": "<|think|>"})

    return messages

  def _parse_thinking_response(
      self, raw_output: str, thinking_enabled: bool
  ) -> tuple[str, Optional[str]]:
    """Parses the raw output to separate thoughts and final answer if enabled."""
    thinking_process = None
    if thinking_enabled:
      thoughts_match = _THOUGHT_PATTERN.search(raw_output)
      if thoughts_match:
        thinking_process = thoughts_match.group(1).strip()
        final_answer = _THOUGHT_PATTERN.sub("", raw_output).strip()
      else:
        final_answer = raw_output.strip()
      text_response = final_answer
    else:
      text_response = raw_output
    return text_response, thinking_process

  def generate_content(
      self,
      contents: List[Union[str, Any]],
      schema: Optional[Any] = None,
      config: Optional[Dict[str, Any]] = None,
      post_process: Optional[Callable[[Any], Any]] = None,
  ) -> Any:
    """Calls LiteLLM completion API.

    Args:
      contents: A list of message parts, which can be strings, `types.Part`
        objects, or dictionaries with "role".
      schema: An optional Pydantic model or dict representing the expected JSON
        schema for the response.
      config: An optional dictionary of configuration parameters to override or
        add to the client's default configuration.
      post_process: An optional callable that takes the parsed JSON object as
        input and returns a processed JSON object. This is called if Pydantic
        validation fails.

    Returns:
      A MockResponse object containing:
        - text: The cleaned text response from the LLM.
        - parsed: The object parsed according to the provided schema, if any.
        - thinking: The extracted thinking process if `enable_thinking` is True.
    """
    merged_config = self.config.copy()
    if config:
      merged_config.update(config)

    supported_params = get_supported_openai_params(model=self.model) or []

    thinking_enabled = merged_config.get(
        "enable_thinking", self.enable_thinking
    )
    temperature = merged_config.get("temperature", self.temperature)

    messages = self._prepare_messages(
        contents, schema, merged_config, thinking_enabled
    )

    completion_args = {
        "model": self.model,
        "messages": messages,
        "temperature": temperature,
        "timeout": self.timeout,
    }

    if self.api_base:
      completion_args["api_base"] = self.api_base
    if self.api_key:
      completion_args["api_key"] = self.api_key
    if schema and "response_format" in supported_params:
      completion_args["response_format"] = schema
    if "extra_body" in merged_config:
      completion_args["extra_body"] = merged_config["extra_body"]

    if self.verbose:
      logging.info("LiteLLM Request: model=%s", self.model)

    def _redact_image_urls(msgs):
      redacted_msgs = copy.deepcopy(msgs)
      for msg in redacted_msgs:
        if isinstance(msg, dict) and isinstance(msg.get("content"), list):
          for content_part in msg["content"]:
            if (
                isinstance(content_part, dict)
                and content_part.get("type") == "image_url"
                and "image_url" in content_part
            ):
              content_part["image_url"] = {"url": "[REDACTED]"}
      return redacted_msgs

    if self.verbose:
      logging.debug(
          "LiteLLM Request: messages=%s", _redact_image_urls(messages)
      )

    def _make_request():
      response = litellm.completion(**completion_args)
      raw_output = response.choices[0].message.content

      text_response, thinking_process = self._parse_thinking_response(
          raw_output, thinking_enabled
      )

      parsed_obj = None
      if schema:
        if (
            not thinking_enabled
            and hasattr(response.choices[0].message, "parsed")
            and response.choices[0].message.parsed
        ):
          parsed_obj = response.choices[0].message.parsed
        else:
          parsed_obj = parse_structured_response(
              text_response, schema, post_process
          )

      usage = None
      if hasattr(response, "usage") and response.usage:
        usage = LLMUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )

      if usage:
        usage_list = token_usage_var.get()
        if usage_list is not None:
          usage_list.append(usage)

      return MockResponse(
          text=text_response,
          parsed=parsed_obj,
          thinking=thinking_process,
          usage=usage,
      )

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
