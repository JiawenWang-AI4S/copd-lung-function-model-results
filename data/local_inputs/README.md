# Local-only prediction inputs

For authorized local figure regeneration, put these files here:

- `Healthy_selected_predictions.csv.gz`
- `COPD_selected_predictions.csv.gz`

Each file must have `task`, `observed`, and `predicted` columns for all six tasks. Participant IDs may be present for local joins but are not required by the plotting script. This directory is excluded from Git. Never commit participant-level predictions or IDs.
