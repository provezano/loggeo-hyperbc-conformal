# Code and benchmark summary data for: Resolving the Stationarity-Outlier-Drift-Finiteness Quadrilemma in Online Conformal Prediction via Log-Geometric Multi-Scale Aggregation

**Author:** Felipe Provezano Coutinho (ORCID: 0000-0001-7206-5856)  
**Version:** 1.0.0 (2026-11-20)  
**License:** MIT (code/) and CC-BY-4.0 (data/)  
**Repository:** <https://github.com/provezano/loggeo-hyperbc-conformal>  
**Archive DOI:** <https://doi.org/10.5281/zenodo.24387339>  

## Contents

- CITATION.cff: Citation metadata with author ORCID.
- LICENSE.txt: MIT and CC-BY-4.0 license terms.
- code/loggeo_hyperbc.py: Reference implementation of LogGeo-HyperBC (Algorithm 1).
- code/verify_proofs.py: Self-contained executable verification of Theorems 1, 2, and 3.
- data/benchmark_summary.json: Aggregated Stage-1 (N = 30) and unseen Stage-2 (N = 30) statistical summaries.

## Quick Verification

```bash
python3 code/verify_proofs.py
```
