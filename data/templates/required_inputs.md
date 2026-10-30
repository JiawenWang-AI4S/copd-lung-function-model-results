# Required local input schema (documentation only)

This file is a schema checklist. It contains no participant rows or genetic dosages.

## Outcomes and split file

`participants_targets_splits.csv.gz` must be created locally from authorized data. Expected columns include a non-public participant key (`eid`), `split` (`train`, `valid`, or `test` for Healthy), `interval_years`, and the six target columns:

- `FEV1_next`, `FVC_next`, `ratio_next`
- `interval_rapid_decline`, `interval_FVC_rapid_decline`, `interval_ratio_rapid_decline`

Missing outcomes should remain missing. Rapid-decline labels must follow the definitions documented in `docs/methods_zh.md`.

## Healthy clinical inputs

The 16 model fields are `Age_BL`, `BMI_BL`, `Height_BL`, `Pack_years_BL`, `FEV1_BL`, `FVC_BL`, `FEV1pred%_BL`, `FEV1_FVC_RATIO_BL`, `Sex`, `SMK_status_BL`, `p30000_i0`, `p30120_i0`, `p30130_i0`, `p30140_i0`, `p30150_i0`, and `interval_years`.

## Genotypes and COPD inputs

Genotype arrays must match the exact authorized variant schema, effect-allele orientation, dosage coding, and genome build used by the original run. The COPD branch also depends on its task-specific clinical feature definitions, 31 lead-SNP fields, saved clinical estimator templates, and fold-local preprocessing objects. These files are deliberately not reconstructed from aggregate results.
