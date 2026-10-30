# COPD and Healthy lung-function prediction models

Public release of the latest comparison figure, aggregate results, model-architecture notes, and research code snapshots.

**This repository does not contain participant-level UK Biobank data, genotype dosages, individual predictions, trained checkpoints, or participant identifiers.**

**Model weights are not included in this release.** Healthy checkpoints were trained on UK Biobank participant-level data. COPD was evaluated with participant-level leave-one-out folds and has no single shared final checkpoint. Before third-party sharing of trained parameters, UK Biobank requires researchers to assess whether a model could retrieve or recreate participant-level information and to satisfy applicable derived-results requirements. No model-specific disclosure review is documented for this release. The public demo therefore uses synthetic values only and does not reproduce reported metrics.

## Study overview

Healthy and COPD identify separate study populations; classification outcomes are rapid-decline events, not COPD diagnosis. The six tasks are next-visit FEV₁, FVC, and FEV₁/FVC plus their corresponding rapid-decline outcomes. See `docs/methods_zh.md` and `docs/model_architecture_zh.md`.

## Repository contents

- `figures/`: six-row, two-column Figure 1 and model architecture diagram. The standalone PR figure has been removed from the current release; AP/AUPRC aggregate values are excluded from this release.
- `results/`, `data/aggregate/`: cohort-level metrics, event counts/prevalence, importance summaries, and contrasts.
- `data/demo/`, `src/synthetic_demo.py`: invented six-task values and a schema-check/demo plotting script.
- `docs/`: Chinese methods, figure captions, architecture, and reproduction notes.
- `src/`: research code snapshots; not a packaged inference application.

## Run the synthetic demo

Requires Python with pandas and matplotlib installed:

```bash
python src/synthetic_demo.py
```

Creates `outputs/synthetic_demo.png`. These invented values only check the demo schema and plotting path. They are not model outputs and cannot be used to reproduce study metrics.

## Reproduce study results

**Paper-performance reproduction is not currently verified by this public release.** The synthetic demo validates input format and plotting only; it is not a real test set. Aggregate tables cannot establish independent reproduction.

Exact model retraining and participant-level curve regeneration require authorized local UK Biobank files, variant schema, preprocessing and clinical-estimator artifacts, and predictions. See `docs/data_access_and_reproduction_zh.md` and `data/templates/required_inputs.md`. Keep all participant-level data and generated checkpoints in approved controlled storage; never commit them here.

The public figures and tables are aggregate research outputs. Confidence intervals are conditional on saved predictions/models unless otherwise stated in the methods.

## License

No reuse license is specified in this release. Public visibility does not itself grant reuse rights.
