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
"""Normalization regular expressions and mappings for RSNA radiology attributes."""

import immutabledict

# Normalization patterns for free-text runtime Modality Type
MODALITY_PATTERNS: immutabledict.immutabledict[str, str] = (
    immutabledict.immutabledict({
        r"\b(cta|ct|computed tomography|cat scan|ct scan)\b": "CT",
        r"\b(mra|mrv|fmri|mrs|mr|mri|magnetic resonance|magnetic resonance imaging|magnetic resonance img)\b": (
            "MR"
        ),
        r"\b(xr|x-ray|radiograph|radiography|plain film|roentgenogram)\b": "XR",
        r"\b(us|usg|ultrasound|ultrasonography|echography|sonogram)\b": "US",
        r"\b(nm|nuclear|scintigraphy)\b": "NM",
        r"\b(pt|pet|pet-ct|pet\/ct)\b": "PT",
        r"\b(rf|fluoroscopy|fluoro)\b": "RF",
        r"\b(dbt|mg|mammogram|mammography)\b": "MG",
    })
)

# Normalization patterns for Modality Subtype
MODALITY_SUBTYPE_PATTERNS: immutabledict.immutabledict[str, str] = (
    immutabledict.immutabledict({
        r"\b(cta|mra|mrv|a|angio|angiography|venography)\b": "Angio",
        r"\b(doppler|color doppler|duplex)\b": "Doppler",
        r"\b(dbt|tomosynthesis|3d mammogram|3d mammography)\b": "tomosynthesis",
        r"\b(spect|single photon)\b": "SPECT",
        r"\b(portable|bedside)\b": "Portable",
        r"\b(fmri|functional)\b": "Functional",
        r"\b(perfusion)\b": "Perfusion",
        r"\b(mrs|spectroscopy)\b": "Spectroscopy",
        r"\b(elastography)\b": "Elastography",
    })
)

# Normalization patterns for Anatomic Region Imaged / Imaging Focus
REGION_PATTERNS: immutabledict.immutabledict[str, str] = (
    immutabledict.immutabledict({
        r"\b(chest|thorax)\b": "Chest",
        r"\b(head|brain|skull)\b": "Head",
        r"\b(abdomen and pelvis|abd and pelvis|abdomen & pelvis|abd & pelvis)\b": (
            "Abdomen and Pelvis"
        ),
        r"\b(abdomen|abd)\b": "Abdomen",
        r"\b(pelvis)\b": "Pelvis",
        r"\b(spine|cervical spine|lumbar spine|thoracic spine|lumbosacral spine)\b": (
            "Spine"
        ),
        r"\b(knee)\b": "Knee",
        r"\b(shoulder)\b": "Shoulder",
        r"\b(hip)\b": "Hip",
        r"\b(breast)\b": "Breast",
        r"\b(extremity upper|upper extremity|arm|wrist|hand|elbow)\b": (
            "Extremity upper"
        ),
        r"\b(extremity lower|lower extremity|leg|ankle|foot)\b": (
            "Extremity lower"
        ),
    })
)

# Normalization patterns for Contrast and Timing
CONTRAST_PATTERNS: immutabledict.immutabledict[str, str] = (
    immutabledict.immutabledict({
        r"\b(w and wo contrast|w & wo contrast|with & without contrast|with and without contrast|wo and w contrast|wo & w contrast)\b": (
            "W and WO contrast"
        ),
        r"\b(w contrast|with contrast|with iv contrast|w\/ contrast|contrast enhanced|cect|cemr)\b": (
            "W contrast"
        ),
        r"\b(wo contrast|without contrast|without iv contrast|wo\/ contrast|non-contrast|noncontrast|ncct)\b": (
            "WO contrast"
        ),
    })
)

# Normalization patterns for View Projection and Aggregation
VIEW_PATTERNS: immutabledict.immutabledict[str, str] = (
    immutabledict.immutabledict({
        r"\b(pa and lateral|pa & lateral|2 views|2 view|ap and lateral|ap & lateral)\b": (
            "PA and Lateral"
        ),
        r"\b(pa view|pa)\b": "PA",
        r"\b(ap view|ap)\b": "AP",
        r"\b(lateral view|lateral|lat)\b": "Lateral",
        r"\b(1 view|single view)\b": "1 View",
        r"\b(3 views|3 view)\b": "3 Views",
        r"\b(4 views|4 view)\b": "4 Views",
    })
)

# Normalization patterns for Laterality
LATERALITY_PATTERNS: immutabledict.immutabledict[str, str] = (
    immutabledict.immutabledict({
        r"\b(left|lt)\b": "Left",
        r"\b(right|rt)\b": "Right",
        r"\b(bilateral|bilat)\b": "Bilateral",
        r"\b(unspecified)\b": "Unspecified",
    })
)
