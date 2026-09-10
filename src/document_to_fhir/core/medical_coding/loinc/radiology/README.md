# Radiology LOINC Coding with RSNA Playbook

This directory contains the offline knowledge base builder and runtime mapper for mapping radiology procedure reports and narrative sections to standardized LOINC codes using the official **RSNA (Radiological Society of North America) / LOINC Radiology Playbook**.

---

## 1. Overview & Architecture

Radiology LOINC mapping uses a structured 6-attribute representation extracted from narrative imaging reports:
1. **Modality Type** (`CT`, `MR`, `XR`, `US`, `MG`, `NM`, `PT`, `RF`)
2. **Modality Subtype** (`Angio`, `Doppler`, `tomosynthesis`, `SPECT`, `Portable`, etc.)
3. **Region Imaged / Imaging Focus** (`Chest`, `Head`, `Abdomen and Pelvis`, `Spine`, `Knee`, etc.)
4. **Contrast Used** (`W contrast`, `WO contrast`, `W and WO contrast`, specific substances)
5. **View Type / Projection & Aggregation** (`PA and Lateral`, `PA`, `AP`, `Lateral`, `1 View`, `2 Views`, etc.)
6. **Laterality** (`Left`, `Right`, `Bilateral`, `Unspecified`)

---

## 2. Multi-Tier Resolution Strategy (`query.py`)

Procedure LOINCs are resolved using a prioritized 3-tier algorithm:

1. **Tier 1 (Exact 6-Tuple Match)**:
   Fast hash lookup on `(Modality, Subtype, Region, Contrast, View, Laterality)`.
2. **Tier 1b (Modality (+ Subtype) + Region + View)**:
   Prioritized for **`XR`** projection studies where view count/projection is the primary discriminator.
3. **Tier 1c (Modality (+ Subtype) + Region + Contrast)**:
   Prioritized for cross-sectional imaging (**`CT`**, **`MR`**) where IV contrast and vascular protocols (`CTA`, `MRA`) are primary discriminators.
4. **Tier 2 (Candidate Filtering + Text Overlap & Rank Scoring)**:
   Filters candidates by modality, subtype, and anatomical region, ranking by token overlap with the exam title and inverse `COMMON_TEST_RANK`.
5. **Tier 3 (Generic Modality Fallback)**:
   Fallback to the top generic study code for the modality or document code `18748-4` (*Diagnostic imaging report*).

---

## 3. How to Build the Offline Knowledge Base Index

The offline builder parses the official multi-row accessory file published by LOINC and RSNA (`LoincRsnaRadiologyPlaybook.csv`), aggregates the concept parts per `LoincNumber`, and writes a single-row indexed CSV.

### Input Files Required
1. **Official RSNA Playbook Accessory CSV**: `LoincRsnaRadiologyPlaybook.csv` (available from [loinc.org/downloads](https://loinc.org/downloads)).
2. *(Optional)* **Master LOINC Table**: `LoincTable.csv` (to read `COMMON_TEST_RANK` for prioritization).

### Running the Builder CLI

<!-- copybara:replace_begin -->
To run the builder in a standard Python environment (simulating open source):

```bash
python -m src.document_to_fhir.core.medical_coding.loinc.radiology.builder_main \
    --input_playbook_csv=/path/to/LoincRsnaRadiologyPlaybook.csv \
    --master_loinc_csv=/path/to/LoincTable.csv \
    --max_common_test_rank=1000 \
    --output_csv=/path/to/destination_playbook_index.csv
```
<!-- copybara:replace_end -->
### Command-Line Arguments
* `--input_playbook_csv` (required): Path to the official `LoincRsnaRadiologyPlaybook.csv` accessory table.
* `--output_csv` (required): Destination path where the compiled single-row KB index CSV should be written.
* `--master_loinc_csv` (optional): Path to `LoincTable.csv` to attach test rank metadata.
* `--max_common_test_rank` (optional): Filter to restrict index to top-N most frequent procedures.

---

## 4. Running Unit Tests

```bash
blaze test //third_party/health_foundations/mddas/v2/src/document_to_fhir/core/medical_coding/loinc/radiology:all
```
