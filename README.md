# TDA-Based Pipeline for Missing Data Imputation Quality Evaluation

This repository provides a **reproducible pipeline** for evaluating the quality of missing-data imputation using **Topological Data Analysis (TDA)**, specifically the **Mapper algorithm** and **permutation-based statistical testing**.

---

## Background

Depression remains one of the leading causes of disability worldwide, highlighting the need for **robust clinical studies** to accurately assess treatment responses.  
Large-scale clinical datasets offer valuable opportunities to understand treatment effectiveness; however, **missing data** presents a major challenge to drawing reliable conclusions.

This project applies **Topological Data Analysis (TDA)** to quantitatively assess and compare different imputation strategies.  
By examining the **topological structures** of complete and imputed datasets, the pipeline evaluates how well the imputed data preserves the intrinsic geometry of the original dataset—an essential step in ensuring the **integrity and reproducibility** of clinical research findings.

---

## Pipeline Overview

The pipeline compares two datasets:
- **Complete-case dataset** – the ground truth (no missing values)
- **Imputed dataset** – after missing-value imputation

It performs:
1. Data standardization and 2D filtering (PCA1 + kNN distance)
2. Mapper graph construction using adaptive K-Means clustering  
3. Computation of topological summaries (connected components, branches, loops)
4. Permutation tests to assess statistical significance of structural differences  
5. Automated report generation (CSV, Markdown, and PKL formats)

---

## Input Format

- Expected input files (CSV format):
complete_case.csv
imputed_data.csv

- Each file must have the same set of feature columns (with matching headers).
- The pipeline automatically handles standardization and filtering.

---

## Quick Start

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

python TDA_pipeline_kmeans_pca.py

```
## Outputs
Summary CSV, Markdown report, and full PKL are saved to:
run_reports/

Temporary caches for parallel computations are stored in:
cache/
Both folders are ignored by Git to keep the repository lightweight.

```

Repository Structure
bash
tda-imputation-mapper/
├── TDA_pipeline_kmeans_pca.py   # Main analysis pipeline
├── requirements.txt             # Dependencies
├── README.md                    # Project documentation
├── .gitignore                   # Ignore cache & results
├── run_reports/ (ignored)       # Output reports
└── cache/ (ignored)             # Temporary computation cache

```

## License
This project is released under the MIT License.


## Contact
Yiyang Ge

Email: yiyang.ge.0@kcl.ac.uk
