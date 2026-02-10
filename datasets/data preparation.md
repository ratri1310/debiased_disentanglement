# Data Preparation Guide

This guide describes how to prepare the biomedical datasets for zero-shot multi-label classification.

## Dataset Descriptions

### Overview
We use three biomedical domain-specific datasets extracted from PubMed:

1. **Neurology Dataset** (`neurology_database.json`)


2. **Immunology Dataset** (`immunology_database.json`)

3. **Embryology Dataset** (`embryology_database.json`)

### Data Source
All datasets are derived from PubMed abstracts with MeSH (Medical Subject Headings) annotations, filtered by domain-specific criteria.

---

## File Format Specifications

### JSON Structure

Each dataset file is a JSON object with the following structure:
```json
{
  "articles": [
    {
      "pmid": "string",
      "title": "string",
      "abstractText": "string",
      "journal": "string",
      "year": "string",
      "meshMajor": ["string", "string", ...],
      "meshMajorEnhanced": [
        {
          "mesh_heading_N": "string",
          "tree_numbers_N": "string",
          "unique_id_N": "string"
        },
        ...
      ]
    },
    ...
  ]
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `pmid` | string | PubMed identifier (unique ID for the article) |
| `title` | string | Article title |
| `abstractText` | string | Full abstract text |
| `journal` | string | Journal name where published |
| `year` | string | Publication year |
| `meshMajor` | array of strings | List of major MeSH terms (simplified labels) |
| `meshMajorEnhanced` | array of objects | Detailed MeSH annotations with hierarchical information |
| `mesh_heading_N` | string | MeSH term name (in enhanced annotations) |
| `tree_numbers_N` | string | MeSH tree numbers (hierarchical classification codes) |
| `unique_id_N` | string | MeSH unique identifier (D-number) |
