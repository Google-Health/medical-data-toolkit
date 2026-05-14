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
"""Tests for model_client.py."""

import datetime
from unittest import mock

from absl.testing import absltest
from google.genai import types
import litellm
import pydantic
import requests

from src.document_to_fhir.common import model_client
from src.document_to_fhir.common.schema import resources


class MockSchema(pydantic.BaseModel):
  reasoning: str


class ModelClientTest(absltest.TestCase):

  @mock.patch("google.genai.Client")
  def test_gemini_client_generate_content(self, mock_genai_client):
    client = model_client.GeminiClient(api_key="fake_key", model="gemini-pro")
    mock_instance = mock_genai_client.return_value

    contents = ["Hello"]
    config = {"temperature": 0.5}
    client.generate_content(contents=contents, config=config)

    expected_config = {
        "response_mime_type": "application/json",
        "temperature": 0.5,
    }
    mock_instance.models.generate_content.assert_called_once_with(
        model="gemini-pro", contents=contents, config=expected_config
    )

  @mock.patch("google.genai.Client")
  def test_gemini_client_verbose(self, unused_mock_genai_client):
    client = model_client.GeminiClient(
        api_key="fake_key", model="gemini-pro", verbose=True)
    self.assertTrue(client.verbose)
    self.assertTrue(client.supports_pdf)

  def test_relaxed_datetime_parsing(self):
    # Test with Patient model which has Optional[datetime.date] dob field
    data = {
        "name": "John Doe",
        "dob": "invalid-date"
    }
    # This should not raise ValidationError, and dob should be None
    patient = resources.Patient.model_validate(data)
    self.assertEqual(patient.name, "John Doe")
    self.assertIsNone(patient.dob)

    # Test with valid date to ensure it still works
    data_valid = {
        "name": "Jane Doe",
        "dob": "1990-01-01"
    }
    patient_valid = resources.Patient.model_validate(data_valid)
    self.assertEqual(patient_valid.dob, datetime.date(1990, 1, 1))

  @mock.patch("litellm.completion")
  @mock.patch(
      "src.document_to_fhir.common.model_client.get_supported_openai_params"
  )
  def test_litellm_client_generate_content(
      self, mock_get_params, mock_completion
  ):
    mock_get_params.return_value = ["response_format"]
    client = model_client.LiteLLMClient(
        model="openai/gemma",
        api_base="http://localhost:8000/v1",
        api_key="fake_key",
    )
    mock_response = mock.Mock()
    mock_choice = mock.Mock()
    mock_message = mock.Mock()
    mock_message.content = '{"reasoning": "litellm reasoning"}'
    mock_message.parsed = None
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_completion.return_value = mock_response

    contents = ["Hello LiteLLM"]
    response = client.generate_content(contents=contents, schema=MockSchema)

    self.assertEqual(response.parsed.reasoning, "litellm reasoning")
    self.assertEqual(mock_completion.call_count, 1)

    _, call_kwargs = mock_completion.call_args
    self.assertEqual(call_kwargs["model"], "openai/gemma")
    self.assertEqual(call_kwargs["api_base"], "http://localhost:8000/v1")
    self.assertEqual(call_kwargs["api_key"], "fake_key")
    self.assertEqual(call_kwargs["response_format"], MockSchema)

  @mock.patch("litellm.completion")
  @mock.patch(
      "src.document_to_fhir.common.model_client.get_supported_openai_params"
  )
  def test_litellm_client_injects_schema_prompt(
      self, mock_get_params, mock_completion
  ):
    mock_get_params.return_value = []
    client = model_client.LiteLLMClient(model="openai/gemma")
    mock_response = mock.Mock()
    mock_choice = mock.Mock()
    mock_message = mock.Mock()
    mock_message.content = '{"reasoning": "ok"}'
    mock_message.parsed = None
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_completion.return_value = mock_response

    contents = ["Hello"]
    client.generate_content(contents=contents, schema=MockSchema)

    _, call_kwargs = mock_completion.call_args
    messages = call_kwargs["messages"]
    self.assertLen(messages, 1)
    self.assertEqual(messages[0]["role"], "user")
    self.assertLen(messages[0]["content"], 2)
    self.assertEqual(messages[0]["content"][0]["text"], "Hello")
    self.assertIn("JSON schema", messages[0]["content"][1]["text"])
    self.assertNotIn("response_format", call_kwargs)

  @mock.patch("litellm.completion")
  @mock.patch(
      "src.document_to_fhir.common.model_client.get_supported_openai_params"
  )
  def test_litellm_client_skips_schema_prompt(
      self, mock_get_params, mock_completion
  ):
    mock_get_params.return_value = ["response_format"]
    client = model_client.LiteLLMClient(model="openai/gemma")
    mock_response = mock.Mock()
    mock_choice = mock.Mock()
    mock_message = mock.Mock()
    mock_message.content = '{"reasoning": "ok"}'
    mock_message.parsed = None
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_completion.return_value = mock_response

    contents = ["Hello"]
    client.generate_content(contents=contents, schema=MockSchema)

    _, call_kwargs = mock_completion.call_args
    messages = call_kwargs["messages"]
    self.assertLen(messages, 1)
    self.assertEqual(messages[0]["role"], "user")
    self.assertLen(messages[0]["content"], 1)
    self.assertEqual(messages[0]["content"][0]["text"], "Hello")
    self.assertEqual(call_kwargs["response_format"], MockSchema)

  @mock.patch("litellm.completion")
  @mock.patch(
      "src.document_to_fhir.common.model_client.get_supported_openai_params"
  )
  def test_litellm_client_handles_parts(self, mock_get_params, mock_completion):
    mock_get_params.return_value = []
    client = model_client.LiteLLMClient(model="openai/gemma")
    mock_response = mock.Mock()
    mock_choice = mock.Mock()
    mock_message = mock.Mock()
    mock_message.content = "ok"
    mock_message.parsed = None
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_completion.return_value = mock_response

    contents = [
        "string part",
        types.Part(text="text part"),
        types.Part(
            inline_data=types.Blob(mime_type="image/png", data=b"imagedata")
        ),
    ]
    client.generate_content(contents=contents)

    _, call_kwargs = mock_completion.call_args
    messages = call_kwargs["messages"]
    self.assertLen(messages, 1)
    self.assertEqual(messages[0]["role"], "user")
    self.assertLen(messages[0]["content"], 3)
    self.assertEqual(messages[0]["content"][0]["text"], "string part")
    self.assertEqual(messages[0]["content"][1]["text"], "text part")
    self.assertEqual(messages[0]["content"][2]["type"], "image_url")
    self.assertIn(
        "data:image/png;base64,", messages[0]["content"][2]["image_url"]["url"]
    )

  @mock.patch("litellm.completion")
  @mock.patch(
      "src.document_to_fhir.common.model_client.get_supported_openai_params"
  )
  def test_litellm_client_handles_dict_messages(
      self, mock_get_params, mock_completion
  ):
    mock_get_params.return_value = []
    client = model_client.LiteLLMClient(model="openai/gemma")
    mock_response = mock.Mock()
    mock_choice = mock.Mock()
    mock_message = mock.Mock()
    mock_message.content = "ok"
    mock_message.parsed = None
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_completion.return_value = mock_response

    contents = [
        {"role": "system", "content": "system instruction"},
        "user prompt",
    ]
    client.generate_content(contents=contents)

    _, call_kwargs = mock_completion.call_args
    messages = call_kwargs["messages"]
    self.assertLen(messages, 2)
    self.assertEqual(messages[0]["role"], "system")
    self.assertEqual(messages[0]["content"], "system instruction")
    self.assertEqual(messages[1]["role"], "user")
    self.assertEqual(messages[1]["content"][0]["text"], "user prompt")

  @mock.patch("litellm.completion")
  @mock.patch(
      "src.document_to_fhir.common.model_client.get_supported_openai_params"
  )
  def test_litellm_client_uses_parsed_response(
      self, mock_get_params, mock_completion
  ):
    mock_get_params.return_value = ["response_format"]
    client = model_client.LiteLLMClient(model="openai/gemma")
    mock_response = mock.Mock()
    mock_choice = mock.Mock()
    mock_message = mock.Mock()
    mock_message.content = '{"reasoning": "ignored"}'
    mock_message.parsed = MockSchema(reasoning="direct parsed")
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_completion.return_value = mock_response

    response = client.generate_content(contents=["Hello"], schema=MockSchema)

    self.assertEqual(response.parsed.reasoning, "direct parsed")

  @mock.patch("time.sleep")
  @mock.patch("litellm.completion")
  @mock.patch(
      "src.document_to_fhir.common.model_client.get_supported_openai_params"
  )
  def test_litellm_client_retry_on_rate_limit(
      self, mock_get_params, mock_completion, mock_sleep
  ):
    mock_get_params.return_value = []
    client = model_client.LiteLLMClient(
        model="openai/gemma", max_retries=3
    )

    mock_response = mock.Mock()
    mock_choice = mock.Mock()
    mock_message = mock.Mock()
    mock_message.content = "ok"
    mock_message.parsed = None
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]

    mock_completion.side_effect = [
        litellm.RateLimitError(
            message="Rate limit",
            response=mock.Mock(headers={}),
            llm_provider="openai",
            model="openai/gemma",
        ),
        mock_response,
    ]

    response = client.generate_content(contents=["Hello"])

    self.assertEqual(response.text, "ok")
    self.assertEqual(mock_completion.call_count, 2)

  @mock.patch("time.sleep")
  @mock.patch("litellm.completion")
  @mock.patch(
      "src.document_to_fhir.common.model_client.get_supported_openai_params"
  )
  def test_litellm_client_retry_on_parsing_error(
      self, mock_get_params, mock_completion, mock_sleep
  ):
    mock_get_params.return_value = []
    client = model_client.LiteLLMClient(
        model="openai/gemma", max_retries=3
    )

    mock_response_bad = mock.Mock()
    mock_choice_bad = mock.Mock()
    mock_message_bad = mock.Mock()
    mock_message_bad.content = "invalid json"
    mock_message_bad.parsed = None
    mock_choice_bad.message = mock_message_bad
    mock_response_bad.choices = [mock_choice_bad]

    mock_response_good = mock.Mock()
    mock_choice_good = mock.Mock()
    mock_message_good = mock.Mock()
    mock_message_good.content = '{"reasoning": "ok"}'
    mock_message_good.parsed = None
    mock_choice_good.message = mock_message_good
    mock_response_good.choices = [mock_choice_good]

    mock_completion.side_effect = [
        mock_response_bad,
        mock_response_good,
    ]

    response = client.generate_content(contents=["Hello"], schema=MockSchema)

    self.assertEqual(response.parsed.reasoning, "ok")
    self.assertEqual(mock_completion.call_count, 2)

  @mock.patch("litellm.completion")
  @mock.patch(
      "src.document_to_fhir.common.model_client.get_supported_openai_params"
  )
  def test_litellm_client_propagates_temperature(
      self, mock_get_params, mock_completion
  ):
    mock_get_params.return_value = []
    client = model_client.LiteLLMClient(
        model="openai/gemma", temperature=0.2
    )

    mock_response = mock.Mock()
    mock_choice = mock.Mock()
    mock_message = mock.Mock()
    mock_message.content = "ok"
    mock_message.parsed = None
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_completion.return_value = mock_response

    client.generate_content(contents=["Hello"])

    self.assertEqual(mock_completion.call_count, 1)
    _, call_kwargs = mock_completion.call_args
    self.assertEqual(call_kwargs["temperature"], 0.2)

  def test_litellm_client_default_temperature(self):
    client = model_client.LiteLLMClient(model="openai/gemma")
    self.assertEqual(client.temperature, 0.0)


if __name__ == "__main__":
  absltest.main()
