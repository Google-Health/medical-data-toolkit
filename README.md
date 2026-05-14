# Medical Data Toolkit

Medical Data Toolkit converts unstructured medical documents (PDFs, images,
screenshots) into accurately structure electronic medical data in the [HL7® FHIR®](https://www.hl7.org/fhir/)
standard format. The toolkit accomplishes this through the use of an LLM model
(external to the tool), specialized document schemas, and advanced extraction
pipelines.

## Vision

Enable information encoded in medical documents to be converted into digitally
accessible in the FHIR®, the interoperability standard for electronic medical
records.

## Key Features

The system achieves high accuracy using the following core modules:

-   **Medical Document Classification**: Identifies the document type (e.g., lab
    report, prescription) to route it to the appropriate processor.
-   **Structured Information Extraction**: Extracts clinical data using
    per-document schemas to help stability and correctness.
-   **Advanced Medical Coding**: Accurately maps concepts to standard LOINC
    codes.
-   **Deterministic FHIR® Conversion**: Rule-based conversion of structured JSON
    into FHIR® R4 resources.

## Terminology & LOINC Mapping

Medical Data Toolkit addresses the high-cardinality challenge of LOINC mapping
for lab reports using a multi-stage strategy:

1.  **Core-Analyte Prediction**: Extracts the primary substance being measured
    (e.g., "Glucose") during initial extraction, achieving near 100% recall.
2.  **Offline Knowledge Base**: Uses pre-computed knowledge base to populate the
    LOINC axes (Property, System, etc.). Constructing this Knowledge Base is a
    prerequisite for offline execution. Detailed instructions can be found in
    the [LOINC
    README](src/document_to_fhir/core/medical_coding/loinc/README.md).
3.  **Signature Matching**: Employs word signatures to handle word-order swaps
    and OCR noise.

These techniques help improve document conversion precision and do not require
external API calls, making the toolkit suitable for offline deployment.

## Usage

The toolkit provides a Docker image which exposes REST API that accepts the raw
bytes of a PDF file or picture of a medical document (i.e., JPEG, PNG) and
returns a completed FHIR® bundle. The docker can be deployed within a serverless
environment (e.g., [Cloud
Run](https://cloud.google.com/run),
[GKE](https://cloud.google.com/kubernetes-engine)), deployed within a [virtual machine](https://cloud.google.com/products/compute),
or executed locally.

### Limitations

Transformation of hand written medical documents to FHIR are not supported.

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

####1. Clone the GitHub Repository

```bash

git clone https://github.com/Google-Health/medical-data-toolkit

```

####2. Build the Medical Data Toolkit Container

Execute from the directory containing the toolkits Dockerfile.

```bash

docker build -t medical-data-toolkit-image .

```

####3. Run the Container

```bash

docker run --name medical-data-toolkit-container -p 8080:8080 -d medical-data-toolkit-image

```

####4. Call the Running Container

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

####5. Stop the Container and Cleanup (Optional)

```bash

docker kill medical-data-toolkit-container && \
docker system prune --all --force

```

## Contributing

We are open to bug reports, pull requests (PR), and other contributions. See
[CONTRIBUTING](CONTRIBUTING.md) for details.

## License

This project is licensed under the Apache 2.0 license, see [LICENSE](LICENSE).
