# Medical Data Toolkit

Medical Data Toolkit converts unstructured medical documents (PDFs, images,
screenshots) into accurately structured electronic medical data in the [HL7® FHIR®](https://www.hl7.org/fhir/)
standard format. The toolkit accomplishes this through the use of an LLM model
(external to the tool), specialized document schemas, advanced extraction
pipelines and medical coding via offline knowledge bases.

## Vision

Enable information encoded in medical documents to be converted into digitally
accessible in the FHIR®, the interoperability standard for electronic medical records.

## Key Features

The system achieves high accuracy using the following core modules:

-   **Medical Document Classification**: Identifies the document type (e.g.,
    lab report, prescription) to route it to the appropriate processor.
-   **Structured Information Extraction**: Extracts clinical data using
    per-document schemas to help stability and correctness.
-   **Advanced Medical Coding**: Accurately maps concepts to standard LOINC and
    SNOMED-CT codes. This currently includes mapping lab tests and radiology
    tests to LOINC codes, prescription medicines to
    [SNOMED CT India Extension drug codes](https://www.nrces.in/services/national-releases/)
    and immunizations to
    [SNOMED CT International Extension vaccination codes](https://nrces.in/ndhm/fhir/r4/ValueSet-ndhm-vaccine-codes.html).
-   **Deterministic FHIR® Conversion**: Rule-based conversion of structured
    JSON into FHIR® R4 resources. This currently supports conversion to
    ABDM-India FHIR profile.

## Terminology & Medical Coding Mapping

Medical Data Toolkit addresses the high-cardinality challenge of medical coding
mapping for laboratory, radiology, prescription and immunization records using
offline multi-stage mapping strategies based on deep domain ontologies.

- LOINC Codes are mapped for tests in laboratory and radiology reports. Refer
[Laboratory LOINC README](src/document_to_fhir/core/medical_coding/loinc/laboratory/README.md)
and
[Radiology LOINC README](src/document_to_fhir/core/medical_coding/loinc/radiology/README.md)
for setting up the knowledge base and more details.

- SNOMED Drug Codes are mapped for medications in Prescriptions. Refer
[SNOMED CT Drug Mapping README](src/document_to_fhir/core/medical_coding/snomed/drug/README.md)
for setting up the knowledge base and more details.

- SNOMED Immunization Codes are mapped for immunizations in Immunization
records. Refer
[SNOMED CT Vaccine Mapping README](src/document_to_fhir/core/medical_coding/snomed/vaccine/README.md)
for setting up the knowledge base and more details.

This helps with working towards high precision and clinical safety without
requiring external API calls, making it suitable for offline, open-source
deployments.

## Usage

The toolkit provides a Docker image which exposes REST API that accepts the raw
bytes of a PDF file or picture of a medical document (i.e., JPEG, PNG) and
returns a completed FHIR® bundle. The docker can be deployed within a serverless
environment (e.g., [Cloud
Run](https://cloud.google.com/run),
[GKE](https://cloud.google.com/kubernetes-engine)), deployed within a [virtual machine](https://cloud.google.com/products/compute),
or executed locally.

### Limitations

- Transformation of hand written medical documents to FHIR® is not supported.
- The toolkit currently supports the following types of medical documents: Diagnostic Report (Laboratory and Radiology Reports), Prescriptions and Immunization Records.
- The toolkit supports FHIR® Implementation Guide for [ABDM](https://nrces.in/ndhm/fhir/r4/profiles.html).

## Toolkit interface Workflow

1.  **Ingest**: Client sends raw bytes of the document.
2.  **Process**: The system classifies the document, extracts data, maps
    terminology, and generates FHIR® resources.
3.  **Respond**: Returns the FHIR® bundle.

The current API is synchronous and optimized for processing small files fast.

## Prerequisites

-   **Models**: Clients can use any LLM model capable of extracting medical data
    from PDF and image files.
-   **Environment**: Serverless container execution environment or Docker.

## Project Structure

-   `src/`: Contains the source code for the server and processing logic.
    -   `rest_server.py`: The Flask entry point.
    -   `document_to_fhir/core/`: Core logic for classification, extraction, and
        FHIR generation.
-   `Dockerfile`: For containerizing the application.

## Getting Started Locally

#### 1. Clone the GitHub Repository

```bash

git clone https://github.com/Google-Health/medical-data-toolkit

```

#### 2. Build Offline Knowledge Bases
Constructing the following Knowledge Bases is required for full terminology mapping functionality:

1. **LOINC (Analyte, System, Property)**: Detailed instructions can be found in the [Laboratory LOINC README](src/document_to_fhir/core/medical_coding/loinc/laboratory/README.md) and [Radiology LOINC README](src/document_to_fhir/core/medical_coding/loinc/radiology/README.md) respectively.

2. **SNOMED CT (Drugs and Vaccines)**: Detailed instructions can be found in [SNOMED CT Drug Mapping README](src/document_to_fhir/core/medical_coding/snomed/drug/README.md) and [SNOMED CT Vaccine Mapping README](src/document_to_fhir/core/medical_coding/snomed/vaccine/README.md) respectively.

#### 3. Build the Medical Data Toolkit Container

Execute from the directory containing the toolkits Dockerfile.

```bash

docker build -t medical-data-toolkit-image .

```

#### 4. Run the Container

```bash

docker run --name medical-data-toolkit-container -p 8080:8080 -d medical-data-toolkit-image

```

#### 5. Call the Running Container

**Example Client Usage (Console)**

```bash

curl -X POST \
     --data-binary @medical_document.pdf \
     "http://127.0.0.1:8080/document_to_fhir"

```

**Example Client Usage (Python)**

```python
import requests

# Assuming the server is running locally or at a specific address
url = "http://127.0.0.1:8080/document_to_fhir"  # Replace with actual endpoint when available

with open("sample_report.pdf", "rb") as f:
  pdf_bytes = f.read()

with requests.post(url, data=pdf_bytes) as response:
  if response.ok:
    print("FHIR Bundle:", response.json())
  else:
    print("Error:", response.text)
```

#### 6. Stop the Container and Cleanup (Optional)

```bash

docker kill medical-data-toolkit-container && \
docker system prune --all --force

```

## License

This project is licensed under the Apache 2.0 license, see [LICENSE](LICENSE).
