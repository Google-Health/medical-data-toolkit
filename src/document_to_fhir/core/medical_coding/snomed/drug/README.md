# SNOMED Drug Matching System

The **SNOMED Drug Matching System** is a modular research framework designed to 
help map clinical medication and prescription details to canonical
[SNOMED-CT India Drug Extension](https://www.nrces.in/services/national-releases/)
codes.

The system operates in two distinct phases:

1.  **Knowledge Base Construction (Offline)**: Builds an offline precomputed
    dataset using the standard tabular data provided by
    [SNOMED-CT India Drug Extension](https://www.nrces.in/services/national-releases/)
    and enriching it by clustering active substances with their equivalent salts
    for improved accuracy.
2.  **Runtime Querying (Online)**: Performs structured retrieval and matching
    against brand, generic, substance, product, dose form, and route indices.

--------------------------------------------------------------------------------

## Codebase Architecture

The project is organized into modular knowledge base builders, offline indices,
and runtime query engines:

### 1. `axes_kb/`

Houses the offline builder logic for derived ontological knowledge bases (such
as substance variations). Each directory typically contains:

-   `builder.py`: Library containing the core logic for constructing the
    knowledge base consisting of active substance clusters.
-   `builder_main.py`: Command-line entry point.

### 2. `index/`

Contains index loaders that read raw or derived SNOMED-CT master files into
memory-efficient DataFrames for search:

-   `brand_index.py`: Index for branded medications (`BrandMaster.txt`).
-   `generic_index.py`: Index for generic drugs (`GenericMaster.txt`).
-   `substance_index.py`: Index for active substances (`SubstanceMaster.txt`).
-   `product_index.py`: Index for drug product names (`ProductMaster.txt`).
-   `substance_variations_index.py`: Index for derived substance variation
    clusters (`SubstanceVariations.txt`).
-   `form_index.py`: Index for dose forms (`DrugFormMaster.txt`).
-   `route_index.py`: Index for routes of administration
    (`RouteOfAdministrationMaster.txt`).

### 3. `query/`

Contains the runtime query engines (`SnomedDrugQueryEngine`, `BrandQuery`,
`GenericQuery`, etc.) that interface with the loaded indices to execute
multi-layered medication matching.

--------------------------------------------------------------------------------

## Getting Started

### Prerequisites

1.  **SNOMED-CT India Drug Extension Dataset**: Obtain the official SNOMED-CT
    Drug Master files from
    [NRCeS National Releases](https://www.nrces.in/services/national-releases)
    from the link: "Get Common Drug Codes for India (Flat Files Package)" (e.g.,
    `SubstanceMaster.txt` and `GenericMaster.txt` tab-separated files).
2.  **Python Version**: Tested with Python 3.11+.

--------------------------------------------------------------------------------

## 1. Knowledge Base Construction (Offline)

The builder iterates over SNOMED substance and generic drug knowledge bases to
cluster active substances with their equivalent salt forms. These substance
variations exist in the `GenericMaster.txt` file. For example, for the generic
name, "Amoxicillin (as amoxicillin sodium) 250 mg powder for solution for
injection vial", we can extract "Amoxicillin" and "Amoxicillin Sodium" as
substance variations using regular expressions. The output of this script is a
TXT file (`SubstanceVariations.txt`) that serves as the foundation for
`SubstanceVariationsIndex`. (All other Index files will be used as is.)

#### Running the Builders

To run the builder in a standard Python environment (simulating open source):

```bash
python -m venv venv
source venv/bin/activate  # Or venv\Scripts\activate on Windows
pip install -r requirements.txt
python -m src.document_to_fhir.core.medical_coding.snomed.drug.axes_kb.substance_variations.builder_main \
  --substance_kb_path="/path/to/SubstanceMaster.txt" \
  --generic_kb_path="/path/to/GenericMaster.txt" \
  --output_folder="/path/to/output/"
```

### Configuration Flags

| Flag                  | Default                   | Description              |
| :-------------------- | :------------------------ | :----------------------- |
| `--substance_kb_path` | (Required)                | Path to the source       |
:                       :                           : SNOMED Substance KB file :
:                       :                           : (`SubstanceMaster.txt`). :
| `--generic_kb_path`   | (Required)                | Path to the source       |
:                       :                           : SNOMED Generic KB file   :
:                       :                           : (`GenericMaster.txt`).   :
| `--output_folder` | (Required)                | Folder path where output |
:                       :                           : CSV will be saved.       :
| `--filename`          | `SubstanceVariations.txt` | Output CSV filename.     |
| `--limit`             | `0`                       | Max number of substances |
:                       :                           : to process (`0` for      :
:                       :                           : all).                    :

--------------------------------------------------------------------------------

## 2. Runtime Querying (Online)

Once the derived knowledge bases are built, use `SnomedDrugQueryEngine` to
perform medication searches and matching.

```python
from document_to_fhir.core.medical_coding.snomed.drug.index import brand_index
from document_to_fhir.core.medical_coding.snomed.drug.index import form_index
from document_to_fhir.core.medical_coding.snomed.drug.index import generic_index
from document_to_fhir.core.medical_coding.snomed.drug.index import product_index
from document_to_fhir.core.medical_coding.snomed.drug.index import route_index
from document_to_fhir.core.medical_coding.snomed.drug.index import substance_index
from document_to_fhir.core.medical_coding.snomed.drug.index import substance_variations_index
from document_to_fhir.core.medical_coding.snomed.drug.query import snomed_drug_query_engine

# 1. Load the search indices
brand_idx = brand_index.BrandIndex.from_index_file("/path/to/BrandMaster.txt")
generic_idx = generic_index.GenericIndex.from_index_file(
    "/path/to/GenericMaster.txt"
)
substance_idx = substance_index.SubstanceIndex.from_index_file(
    "/path/to/SubstanceMaster.txt"
)
product_idx = product_index.ProductIndex.from_index_file(
    "/path/to/ProductMaster.txt"
)
substance_variations_idx = (
    substance_variations_index.SubstanceVariationsIndex.from_index_file(
        "/path/to/SubstanceVariations.txt"
    )
)
form_idx = form_index.FormIndex.from_index_file("/path/to/DrugFormMaster.txt")
route_idx = route_index.RouteIndex.from_index_file(
    "/path/to/RouteOfAdministrationMaster.txt"
)
print("Loaded SNOMED Drug KBs.")

# 2. Initialize the query engine
engine = snomed_drug_query_engine.SnomedDrugQueryEngine(
    brand_idx=brand_idx,
    generic_idx=generic_idx,
    substance_idx=substance_idx,
    product_idx=product_idx,
    substance_variations_idx=substance_variations_idx,
    form_idx=form_idx,
    route_idx=route_idx,
)
```
