# SNOMED Vaccine Matching System

The **SNOMED Vaccine Matching System** is a modular research framework designed to
help map vaccines in an immunization record to canonical
[SNOMED-CT International Edition](https://nrces.in/ndhm/fhir/r4/ValueSet-ndhm-vaccine-codes.html) codes (`ndhm-vaccine-codes` ValueSet).

The system operates in two distinct phases:

1.  **Knowledge Base Construction (Offline)**: Builds an offline precomputed,
    enriched dataset using the standard tabular data provided by the
    [NRCeS NDHM FHIR Vaccine ValueSet](https://nrces.in/ndhm/fhir/r4/ValueSet-ndhm-vaccine-codes.html)
    (179 vaccine concepts) and enriching it using a sequential 3-stage LLM
    pipeline to extract parent-child hierarchies, comprehensive clinical
    variations/brand names, and active antigen constituents, followed by
    deterministic post-processing.
2.  **Runtime Querying (Online)**: Performs structured retrieval and matching
    against the enriched vaccine index (`SnomedVaccineIndex`) using
    `SnomedVaccineQueryEngine`, normalizing clinical text tokens, matching
    against clinical variations and antigens, resolving matches via patient age
    categorization (Adult vs. Paediatric vs. Unknown), and traversing concept
    hierarchy graphs to identify combination vaccines or top parent codes.

--------------------------------------------------------------------------------

## Codebase Architecture

The project is organized into modular knowledge base builders, offline indices,
runtime query engines, and utilities:

### 1. `axes_kb/`

Houses the offline builder logic for the derived vaccine ontological knowledge
base using LLMs:

-   `builder.py`: Core library implementing `VaccinesKBBuilder`, which
    orchestrates the sequential 3-stage LLM generation pipeline and
    post-processing.
-   `builder_main.py`: Command-line entry point to run the pipeline with
    configurable LLM clients (Gemini, Gemma, LiteLLM).
-   `prompts.py`: Prompt templates used across the 3 sequential LLM stages
    (`PROMPT_SNOMED_VACCINES_PARENT_CHILD_HIERARCHY`,
    `PROMPT_SNOMED_VACCINES_CLINICAL_VARIATIONS`,
    `PROMPT_SNOMED_VACCINES_ANTIGENS`).
-   `builder_test.py`: Unit tests for builder pipeline logic, prompt formatting,
    LLM response parsing, and post-processing validation.

### 2. `index/`

Contains index loaders that read raw or enriched SNOMED-CT vaccine master files
into memory-efficient DataFrames for search:

-   `snomed_vaccine_index.py`: Index for SNOMED-CT vaccines
    (`SnomedVaccineIndex`), which loads the enriched TSV file into memory,
    precomputes normalized clinical variations and antigens
    (`norm_clinical_variations`), and parses patient demographics
    (`patient_type`).
-   `snomed_vaccine_index_test.py`: Unit tests for index loading, missing file
    errors, preprocessing, and patient type parsing.

### 3. `query/`

Contains the runtime query engines and protocols that interface with the loaded
vaccine index to execute clinical matching:

-   `base.py`: Defines the `VaccineResult` Pydantic model (`code`,
    `display_name`, `code_system`) and abstract base classes
    (`BaseVaccineQueryEngine`, `BaseVaccineQuery`).
-   `snomed_vaccine_query_engine.py`: Implements `SnomedVaccineQueryEngine`,
    which normalizes candidate terms, searches index variations using regex
    token boundaries, filters out overly broad SNOMED concepts, resolves matches
    based on patient type, and resolves combination vaccines via parent
    traversal.
-   `snomed_vaccine_query_engine_test.py`: Unit tests covering token matching,
    patient type filtering, combination vaccine disambiguation, and edge cases.

### 4. `utils.py`

Shared helper utilities for vaccine string processing and hierarchical graph
traversal:

-   `remove_vaccine_keywords`: Strips common boilerplate keywords (e.g.,
    "vaccine", "vax", "antigen", "virus", "bacteria", "pediatric", "adult").
-   `parse_and_normalize_vaccines`: Expands parenthesized subterms, strips
    punctuation, whitespace, and hyphens, and extracts normalized vaccine tokens
    (e.g., handling combination vaccines like "DTaP-Hib").
-   `get_top_parent_code`: Traverses parent-child relationships upward to
    identify top-level non-excluded parent concepts or combination vaccine
    ancestors.
-   `utils_test.py`: Unit tests validating keyword stripping, normalization
    rules, and ancestor graph traversal.

--------------------------------------------------------------------------------

## Getting Started

### Prerequisites

1.  **SNOMED-CT Vaccine Dataset**: Obtain the official SNOMED-CT NDHM Vaccine
    Codes ValueSet from the
    [FHIR Implementation Guide for ABDM](https://nrces.in/ndhm/fhir/r4/ValueSet-ndhm-vaccine-codes.html#expansion).
    \
    Copy over the table provided in the above URL (containing the SNOMED
    concept codes and their corresponding preferred display terms) into a CSV
    file (e.g. `snomed_vaccine_raw.csv`).
2.  **LLM API Access**: Ensure you have an API key configured for the LLM
    client (e.g., `GEMINI_API_KEY` or `API_KEY` for Gemini, or credentials for
    Gemma/LiteLLM).
3.  **Python Version**: Tested with Python 3.11+.

### Setup

Configure your environment with an API key:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

--------------------------------------------------------------------------------

## 1. Knowledge Base Construction (Offline)

The builder constructs the enriched vaccine knowledge base using a sequential
3-stage LLM pipeline at temperature = 0:

1.  **Stage 1 (Parent-Child Hierarchy)**: Prompts the LLM to identify direct
    parent SNOMED codes (`parent_snomed_code`) for each of the 179 vaccine
    concepts, establishing the ontological hierarchy.
2.  **Stage 2 (Clinical Variations & Brands)**: Generates an exhaustive list of
    clinical aliases (`clinical_variations`) across 4 layers:
    -   Official public health acronyms and abbreviations (CDC, WHO, UIP, e.g.,
        bOPV, DTwP, MMR, JE).
    -   Clinical shorthand and EHR slang (e.g., 'Flu shot', 'TB vax', 'Painless
        DPT', 'Triple antigen').
    -   Indian market trade names (Serum Institute of India, Bharat Biotech,
        Biological E, Panacea Biotec, etc., e.g., Covishield, Covaxin, Rotasiil,
        Typbar-TCV, Easyfive).
    -   Prominent international trade names (approved by FDA/EMA/WHO).
3.  **Stage 3 (Antigen Constituents)**: Extracts active pathogen antigens and
    toxoids (`antigens`) formatted as `[Organism] antigen` or `[Organism]
    toxoid`, delimited by ` + `.
4.  **Post-Processing**: Resolves parent preferred display terms
    (`parent_preferred_display_term`) via SNOMED code lookup and derives patient
    demographic categories (`patient_type`: `ADULT`, `PAEDIATRIC`, or `UNKNOWN`)
    from the concept names.

The final enriched dataset is saved as a tab-separated file (TSV) that serves as
the foundation for `SnomedVaccineIndex`.

#### Running the Builders

To run the builder in a standard Python environment (simulating open source):

```bash
python -m venv venv
source venv/bin/activate  # Or venv\Scripts\activate on Windows
pip install -r src/document_to_fhir/core/medical_coding/snomed/requirements.txt
python -m src.document_to_fhir.core.medical_coding.snomed.vaccine.axes_kb.builder_main \
  --input_csv_path="/path/to/snomed_vaccine_raw.csv" \
  --input_snomed_code_col_name="snomed_code" \
  --input_preferred_term_col_name="preferred_display_term" \
  --output_tsv_path="/path/to/output/SnomedVaccineIndex.tsv" \
  --client_type="gemini" \
  --model_name="gemini-3.6-flash"
```

### Configuration Flags

| Flag                             | Default                      | Description                                                                            |
| :------------------------------- | :--------------------------- | :------------------------------------------------------------------------------------- |
| `--input_csv_path`               | (Required)                   | Path to the source input CSV containing SNOMED vaccine codes and preferred terms.      |
| `--input_snomed_code_col_name`   | (Required)                   | Name of the column in the input CSV containing the SNOMED code.                        |
| `--input_preferred_term_col_name` | (Required)                  | Name of the column in the input CSV containing the preferred display term.             |
| `--output_tsv_path`              | (Required)                   | Filepath where the final post-processed TSV will be saved.                             |
| `--client_type`            | (Required)                   | Type of LLM client to use (`gemini`, `gemma`, `litellm`).                              |
| `--model_name`             | (Required)                   | LLM model name.                                                                        |
| `--api_key`                | (`$API_KEY` / `$GEMINI_API_KEY`) | API Key for the LLM client. Defaults to environment variables.                         |
| `--max_attempts`           | `3`                          | Maximum retry attempts per pipeline stage on failure.                                  |

--------------------------------------------------------------------------------

## 2. Runtime Querying (Online)

Once the derived knowledge base is built, use `SnomedVaccineIndex` and
`SnomedVaccineQueryEngine` to perform vaccine searches and matching.

```python
from document_to_fhir.common.schema import resources
from document_to_fhir.core.medical_coding.mapper.terminology_mappers import snomed_vaccine_terminology_mapper
from document_to_fhir.core.medical_coding.snomed.vaccine.index import snomed_vaccine_index
from document_to_fhir.core.medical_coding.snomed.vaccine.query import snomed_vaccine_query_engine

# 1. Load the search index
vaccine_idx = snomed_vaccine_index.SnomedVaccineIndex.from_index_file(
    "/path/to/SnomedVaccineIndex.tsv"
)
print("Loaded SNOMED Vaccine KB.")

# 2. Initialize the query engine
engine = snomed_vaccine_query_engine.SnomedVaccineQueryEngine(
    vaccine_index=vaccine_idx
)

# 3. Direct Querying:
# Match a vaccine name string and patient demographic context
immunization = resources.Immunization(
    vaccine_name="Covishield",
    patient_type=resources.PatientType.ADULT,
    dosage="1",
)
result = engine.query(immunization)
print(f"Matched Code: {result.code}")
print(f"Display Name: {result.display_name}")
print(f"Code System: {result.code_system}")

# 4. Or use via the Terminology Mapper:
mapper = snomed_vaccine_terminology_mapper.SnomedVaccineTerminologyMapper(
    query_engine=engine
)
mapper.map_inplace(immunization)
print(f"Mapped Immunization Vaccine Code: {immunization.vaccine_code}")
print(f"Mapped Immunization Display Name: {immunization.vaccine_code_name}")
```
