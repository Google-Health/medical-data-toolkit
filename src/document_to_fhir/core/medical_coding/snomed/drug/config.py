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
"""Configuration settings for SNOMED drug matching."""

# Global flags
# Whether to enable matching for brand drugs.
ENABLE_BRAND = True
# Whether to enable matching for generic drugs.
ENABLE_GENERIC = True
# Whether to enable search tool based matching.
ENABLE_SEARCH_TOOL = False

# Brand matching features
# If True, require strength to for brand matching.
BRAND_MATCHING_STRENGTH = True
# If True, enable unit conversion for strength matching.
BRAND_MATCHING_ENABLE_UNIT_CONVERSION = True
# If True, require dose form to match.
BRAND_MATCHING_DOSE_FORM = True
# If True, require route of administration to match.
BRAND_MATCHING_ROUTE = False

# Generic matching features
# If True, support alternate substances for generic matching.
GENERIC_MATCHING_SUPPORT_ALTERNATE_SUBSTANCES = True
# If True, require strength to match.
GENERIC_MATCHING_STRENGTH = True
# If True, enable unit conversion for strength matching.
GENERIC_MATCHING_ENABLE_UNIT_CONVERSION = True
# If True, require dose form to match.
GENERIC_MATCHING_DOSE_FORM = True
# If True, require route of administration to match.
GENERIC_MATCHING_ROUTE = False

# Search tool matching features
# If True, support alternate substances for search tool matching.
SEARCH_TOOL_MATCHING_SUPPORT_ALTERNATE_SUBSTANCES = True
# If True, require strength to match for search tool matching.
SEARCH_TOOL_MATCHING_STRENGTH = True
# If True, require dose form to match for search tool matching.
SEARCH_TOOL_MATCHING_DOSE_FORM = True
# If True, require route to match for search tool matching.
SEARCH_TOOL_MATCHING_ROUTE = True
# If True, enable unit conversion for search tool strength matching.
SEARCH_TOOL_MATCHING_ENABLE_UNIT_CONVERSION = True
