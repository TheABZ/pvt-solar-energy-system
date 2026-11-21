# Hybrid Photovoltaic–Thermal Solar Energy System

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24419431.svg)](https://doi.org/10.5281/zenodo.24419431)

Computational modeling and efficiency optimization by **Gabriel Chiazor Unomah**, B.Sc. Physics, University of Lagos.

This repository contains the Python source and supporting numerical records accompanying the PV/T thesis prepared for academic review. The source was recovered from the latest review package; the baseline model is unchanged.

## What the code studies

| Study | Purpose | Interpretation |
| --- | --- | --- |
| Controlled pure-water simulation | Solve cell, backplate and water temperatures; compare dry and cooled panels; sweep water flow while accounting for an assumed pump law | Illustrative parameters and synthetic sunlight; the best tested flow depends on these assumptions |
| Conditional experimental benchmark | Fit and evaluate a three-state model using published water–glycol collector measurements | Measured weather, inlet, flow, rear-surface and sky conditions are supplied as boundaries; this does not measure a dry-versus-cooled benefit |

## Quick start — Windows, macOS or Linux

Install Python 3.10 or newer, then download this repository using **Code → Download ZIP** and extract it. Open a terminal in the extracted folder.

```sh
python -m pip install -r requirements.txt
python baseline_model.py
python figures_and_audit.py
python validation_tools.py
```

On Windows, use `py` if `python` is unavailable. On some Linux/macOS installations use `python3`.

The baseline writes `results.json`, `pvt_results.csv` and `pvt_comparison.png` beside the script. It also reports an open-rack dry reference (rear loss equal to the front loss) alongside the rear-insulated dry panel. The baseline also reports a dry/cooled comparison, a finite flow sweep, an energy balance residual, solver refinement, darkness equilibrium and a zero temperature-coefficient check. `figures_and_audit.py` regenerates the paper's simulation figures (Figs. 3–7) at 300 dpi and writes `figures/audit_results.json` with the noon heat rates, daily energy accounting and Jacobian time scales. `validation_tools.py` checks units, heat-flow signs and conservation including stagnant water.

## Experimental workflow

```sh
python prepare_validation.py --download
python replay_frozen_model.py
```

The downloader retrieves approximately 205 MiB of public CSV data, verifies its checksum and writes quality summaries to `outputs/`. Allow roughly 1 GB of available RAM. The raw data are not redistributed here. The replay uses the saved parameters and pre-test forecast state in `experimental_outputs/frozen_model.json`. It checks the first held-out date; it does not repeat calibration or compute all reserved-period scores.

To repeat the complete calibration and reserved-period evaluation:

```sh
python experimental_benchmark.py
```

This step requires Linux, macOS or Windows Subsystem for Linux, with a C compiler available as `cc`. It compiles `thermal_core.c` locally to accelerate repeated integration. Ordinary Windows Python can run the baseline and portable replay without a compiler. Experimental scripts overwrite their corresponding result records when rerun.

## Saved research records

`experimental_outputs/benchmark_results.json` contains the original full-period fits and metrics. `frozen_model.json` stores the selected parameters and forecast states; `portable_replay_results.json` stores the earlier first-date replay. These are carried over from the reviewed package. A full rerun of the experimental workflow from the public download (21 November 2026: checksum verified, 1,002,240 records, calibration and reserved-period evaluation repeated) reproduced the saved benchmark records to floating-point precision.

The original package reports these reserved-period RMSE values:

| Output | RMSE |
| --- | ---: |
| PV temperature | 1.249853 °C |
| Flowing outlet temperature | 1.080775 °C |
| Filtered electrical power | 6.086770 W |

Two fitted parameters reach their declared bounds. Parameter uniqueness, confidence intervals and sensor uncertainties remain unresolved. No paired dry-panel reference or measured pump power is available. The conditional benchmark therefore does not experimentally establish cooling gains or a pump optimum. This project remains subject to academic review.

## Files

| File or folder | Contents |
| --- | --- |
| `baseline_model.py` | Commented illustrative model, flow sweep, open-rack dry reference and plots |
| `figures_and_audit.py` | Paper simulation figures (Figs. 3–7) and audit values (Tables 6 and 8, Jacobian time scales) |
| `validation_tools.py` | Unit, sign and energy balance checks |
| `prepare_validation.py`, `field_mapping.csv` | Public data retrieval, checksum checks and measured-field mapping |
| `experimental_benchmark.py`, `thermal_core.c` | Calibration and chronological held-out evaluation |
| `replay_frozen_model.py` | Independent SciPy integration with frozen parameters |
| `experimental_outputs/` | Original fitted model, scores and daily error records |
| `data_quality/` | Original source quality summaries and date partitions |
| `numerical_audit/` | Earlier detailed numerical verification records |
| `source_manifest.json`, `environment.json` | Source provenance and original package versions |

The thesis PDF, editable thesis and private reviewer correspondence are supplied separately from this code repository.

## How to cite

If you use this code, please cite the archived release: Unomah, G. C. (2026). *Hybrid photovoltaic–thermal solar energy system: simulation and experimental benchmark code*. Zenodo. https://doi.org/10.5281/zenodo.24419431 (version 1.0.0). To cite whichever version is latest, use the concept DOI https://doi.org/10.5281/zenodo.24419430. A `CITATION.cff` file is included for citation managers.

## Data sources

- Public dataset: [DOI 10.57739/3720](https://doi.org/10.57739/3720)
- Data article: [DOI 10.1016/j.dib.2023.109417](https://doi.org/10.1016/j.dib.2023.109417)
- Related model article: [DOI 10.1016/j.enbuild.2023.113277](https://doi.org/10.1016/j.enbuild.2023.113277)

Consult these sources for their data terms and cite the original authors when using their measurements. The repository's MIT licence applies to the project code; external data retain their original terms.
