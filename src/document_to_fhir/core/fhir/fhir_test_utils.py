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
"""Utility functions for FHIR tests."""

from typing import Any

from absl.testing import absltest
from google.protobuf import message


def normalize_fhir_bundle(bundle: message.Message):
  """Normalizes a FHIR bundle for comparison in tests.

  Clears fields that are non-deterministic (like UUIDs in 'id', 'fullUrl',
  and 'uri' fields) or may have different representations of the same value
  (like 'Z' vs 'UTC' for timezone).

  Args:
    bundle: A FHIR bundle proto to normalize.
  """
  _clear_nested_fields(
      bundle, ['id', 'full_url', 'uri', 'timezone', 'version_id', 'date']
  )


def _clear_nested_fields(proto_message, field_names):
  """Recursively clears nested fields by name."""
  for field_descriptor, value in proto_message.ListFields():
    if field_descriptor.name in field_names:
      proto_message.ClearField(field_descriptor.name)
      continue
    if field_descriptor.type == field_descriptor.TYPE_MESSAGE:
      if field_descriptor.label == field_descriptor.LABEL_REPEATED:
        for item in value:
          _clear_nested_fields(item, field_names)
      else:
        _clear_nested_fields(value, field_names)


def _find_references(data: Any, refs: list[str]):
  """Finds all 'urn:uuid:' references in a FHIR resource."""
  if isinstance(data, list):
    for item in data:
      _find_references(item, refs)
  elif isinstance(data, dict):
    if (
        'reference' in data
        and isinstance(data['reference'], str)
        and data['reference'].startswith('urn:uuid:')
    ):
      refs.append(data['reference'])
    for value in data.values():
      if isinstance(value, (dict, list)):
        _find_references(value, refs)


def assert_fhir_references_valid(
    test_case: absltest.TestCase, bundle: dict[str, Any]
):
  """Asserts that all inter-resource references within the bundle are valid."""
  full_urls = set()
  for entry in bundle.get('entry', []):
    if (
        'fullUrl' in entry
        and entry['fullUrl']
        and entry['fullUrl'].startswith('urn:uuid:')
    ):
      full_urls.add(entry['fullUrl'])

  bundle_refs = []
  for entry in bundle.get('entry', []):
    if 'resource' in entry:
      _find_references(entry['resource'], bundle_refs)

  invalid_refs = []
  for ref in bundle_refs:
    if ref not in full_urls:
      invalid_refs.append(ref)

  test_case.assertEmpty(
      invalid_refs,
      f'Bundle contains invalid references to other resources: {invalid_refs}',
  )
