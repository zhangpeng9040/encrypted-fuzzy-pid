# Paper-to-file mapping

| Manuscript result | Retained files | Analysis |
|---|---|---|
| Table 6, Section 6.2 | `data/partition_comparison/`, `coefficients/partitions/`, `configs/partitions.json` | `scripts/recompute_results.py`, `scripts/check_approximation.py` |
| Table 7 / Figure 3 | `data/pid_plain/`, `data/ckks_pid/` | `scripts/recompute_results.py`, `scripts/plot_results.py` |
| Table 8 / Figure 4 | `data/fuzzy_comparison/`, chen and fuzzy modes | Same |
| Table 9 / Figure 5 | `data/fuzzy_comparison/`, fuzzy/poly/ckks_poly modes | Same |
| Table 10 / Figure 6 | `data/ckks_pid/`, `data/fuzzy_comparison/ckks_poly_trajectory.csv` | Same |
| Section 6.5 error decomposition | `data/control_output_errors/`, original encrypted trajectory | `scripts/check_error_decomposition.py` |
| Table 12, Section 6.7 | `data/operating_conditions/` | `scripts/recompute_results.py` |

Tables 1–5 describe the methods and parameters implemented in `code/`, documented in `docs/PARAMETER_SOURCES.md`; partition definitions are in `configs/partitions.json`. Table 11 is a literature comparison, supported by `docs/references.bib`.

The selected coefficient array `coefficients/piecewise_coefficients.npy` is the 384-cell approximation used by the plaintext and encrypted polynomial controllers. The partition experiment retains only the four configurations reported in the main text. Frozen data remain full precision.
