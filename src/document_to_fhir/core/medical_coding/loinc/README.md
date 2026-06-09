# LOINC Prediction System

The **LOINC Prediction System** is a modular research framework designed to help
map clinical lab test details to canonical [LOINC](https://loinc.org/) codes.
It uses LLMs to perform semantic enrichment of LOINC structured knowledge bases
to enable accurate, context-aware predictions.

The system operates in two distinct phases:

1.  **Knowledge Base Construction (Offline)**: Builds an offline precomputed
    dataset by enriching standard LOINC data with predicted core analytes and
    synonyms using LLMs.
2.  **Runtime Querying (Online)**: Performs multi-layered retrieval based on analyte signatures and specimen-type filtering.

---

## Codebase Architecture

The project is organized into modular "axes" and shared infrastructure:

### 1. `axes_kb/`
Houses the individual logic buckets for various ontological vectors
(core_analyte, system, property, scale_type, etc.). Each directory typically
contains:

- `builder.py`: Library containing the core logic to build the Knowledge Base.
- `builder_main.py`: Command-line entry point.
- `prompt.py`: Contains LLM prompt templates.
- `mapper.py` or `index.py`: Logic for mapping/indexing terms.

### 2. Main API for Querying

- [`query.py`](query.py): The runtime engine that interfaces with indices to
  execute multi-layered retrieval.

---

## Getting Started

### Prerequisites
1.  **LOINC Dataset**: Download the official LOINC Table (CSV) from
    [loinc.org](https://loinc.org/downloads/loinc-table/).
2.  **API Access**: Ensure you have a valid LLM API key.
3.  **Python Version**: Tested with Python 3.13.12.

### Setup
Configure your environment:
```bash
export API_KEY="your_api_key_here"
```

---

## 1. Knowledge Base Construction (Offline)

The builder iterates over a filtered LOINC dataset and uses LLMs to predict
canonical features for different axes (Core Analyte, System, Property). It
writes enriched CSVs that serve as the foundation for the search index and
mappers.

#### Running the Builders

<!-- copybara:replace_begin -->
To run the library in a standard Python environment (simulating open source):

```bash
python -m venv venv
source venv/bin/activate  # Or venv\Scripts\activate on Windows
pip install -r src/document_to_fhir/core/medical_coding/loinc/requirements.txt
python -m src.document_to_fhir.core.medical_coding.loinc.axes_kb.<axis_name>.builder_main \
  --loinc_csv_path="/path/to/input/LoincTable.csv" \
  --output_csv_folder="/path/to/output/folder/" \
  --client_type="litellm" \
  --model_name="gemma-4-26b-a4b-it" \
  --max_rank=2000 \
  --workers=10
```

Where `<axis_name>` corresponds to the axis you want to build
(e.g., `core_analyte`, `system`, `property`).
<!-- copybara:replace_end -->

### Configuration Flags
| Flag | Default | Description |
| :--- | :--- | :--- |
| `--loinc_csv_path` | (Required) | Path to the source LOINC CSV. |
| `--output_csv_folder` | (Required) | Folder where enriched CSVs are saved. |
| `--client_type` | (Required) | Type of LLM client to use (gemini, gemma, litellm). |
| `--api_key` | (Required) | Your LLM API key. |
| `--model_name` | (Required) | The model identifier for inference. |
| `--max_rank` | `2000` | Process only common tests (based on `COMMON_TEST_RANK`). |
| `--limit` | `0` | Max rows to process (use `0` for all). |
| `--workers` | `10` | Number of parallel threads for enrichment. |

---

## 2. Runtime Querying (Online)

Once the Knowledge Base is built, use the `LoincQueryEngine` to perform searches.

```python
from document_to_fhir.common.schema import resources
from document_to_fhir.core.medical_coding.loinc import query
from document_to_fhir.core.medical_coding.loinc.axes_kb.core_analyte import index
from document_to_fhir.core.medical_coding.loinc.axes_kb.system import mapper as system_mapper_lib
from document_to_fhir.core.medical_coding.loinc.axes_kb.scale_type import mapper as scale_mapper_lib
from document_to_fhir.core.medical_coding.loinc.axes_kb.property import mapper as property_mapper_lib

# 1. Load the search index and mappers
analyte_index = index.AnalytesIndex.from_csv("/path/to/enriched_analyte_kb.csv")
system_mapper = system_mapper_lib.SpecimenToSystemMapper.from_csv("/path/to/system_kb.csv")
scale_mapper = scale_mapper_lib.ScaleMapper()
property_mapper = property_mapper_lib.UnitToPropertyMapper.from_csv("/path/to/property_kb.csv")
print("Loaded Offline KBs.")

# 2. Initialize the query engine with all filters
engine = query.LoincQueryEngine(
    analyte_index=analyte_index,
    system_mapper=system_mapper,
    scale_mapper=scale_mapper,
    property_mapper=property_mapper
)

# 3. Execute a query
lab_test = resources.LabTest(
    core_analyte="Glucose",
    specimen="urine",
    name="Glucose Random",
    result="100.5",
    unit="mg/dL"
)
results = engine.query(lab_test)

for rec in results:
    print(f"LOINC: {rec.loinc_num} | Name: {rec.long_common_name}")
```

