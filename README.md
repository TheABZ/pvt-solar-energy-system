# Hybrid Photovoltaic–Thermal Solar Energy System

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24419430.svg)](https://doi.org/10.5281/zenodo.24419430)

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

To estimate what water cooling would achieve on the measured collector (weather-driven model, about 30 s):

```sh
python weather_driven_counterfactual.py
```

This refits the same three-state model with ambient air as the rear boundary (the collector has no rear insulation), so it needs only weather, inlet temperature and flow. It is scored once on the reserved days and then compares a dry module, the measured operation and ambient-temperature inlet at several flows over all 58 measured days. With the fitted electrical law, the best net electrical gain is 0.47% (20 L/h); with a typical -0.4 %/K coefficient it is 1.09%. Results are written to `experimental_outputs/weather_model.json`, `counterfactual_results.json` and `weather_counterfactual.png`.

To reproduce the closed-form coupling factor, cooling effectiveness and critical inlet temperature (paper Section 2.14, Table 14 and Fig. 11):

```sh
python cooling_effectiveness.py
```

Eliminating the water and backplate nodes in the quasi-steady limit gives the cell cooling as `F_c * (T_b,dry - T_in)`, where `F_c` is a cell-to-fluid coupling factor and `T_b,dry` is the backplate temperature of the dry module. Circulation therefore stops cooling once the inlet reaches the dry backplate temperature, at any flow. The script checks the closed form against the transient model (agreement within 0.012 in effectiveness), holds the inlet at the critical temperature to confirm that cooling vanishes, and applies the result to the measured collector if `experimental_outputs/weather_model.json` is present. Results go to `experimental_outputs/cooling_effectiveness.json` and `figures/fig11_cooling_effectiveness_map.png`.

The two experimental fitting steps above require Linux, macOS or Windows Subsystem for Linux, with a C compiler available as `cc`. It compiles `thermal_core.c` locally to accelerate repeated integration. Ordinary Windows Python can run the baseline and portable replay without a compiler. Experimental scripts overwrite their corresponding result records when rerun.

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
| `weather_driven_counterfactual.py` | Weather-driven model of the measured collector and cooling scenarios |
| `cooling_effectiveness.py` | Closed-form coupling factor, cooling effectiveness and critical inlet temperature (Fig. 11) |
| `experimental_outputs/` | Original fitted model, scores and daily error records; weather-driven model and cooling scenarios |
| `data_quality/` | Original source quality summaries and date partitions |
| `numerical_audit/` | Earlier detailed numerical verification records |
| `source_manifest.json`, `environment.json` | Source provenance and original package versions |

The thesis PDF, editable thesis and private reviewer correspondence are supplied separately from this code repository.

## How to cite

If you use this code, please cite the archived release: Unomah, G. C. (2026). *Hybrid photovoltaic–thermal solar energy system: simulation and experimental benchmark code*. Zenodo. Concept DOI (always resolves to the latest version): https://doi.org/10.5281/zenodo.24419430. Version 1.0.0: https://doi.org/10.5281/zenodo.24419431. Version 1.1.0 (weather-driven cooling scenarios): https://doi.org/10.5281/zenodo.24419952. Version 1.2.0 adds the closed-form critical inlet temperature analysis; its version DOI is listed on the Zenodo record. A `CITATION.cff` file is included for citation managers.

## Data sources

- Public dataset: [DOI 10.57739/3720](https://doi.org/10.57739/3720)
- Data article: [DOI 10.1016/j.dib.2023.109417](https://doi.org/10.1016/j.dib.2023.109417)
- Related model article: [DOI 10.1016/j.enbuild.2023.113277](https://doi.org/10.1016/j.enbuild.2023.113277)

Consult these sources for their data terms and cite the original authors when using their measurements. The repository's MIT licence applies to the project code; external data retain their original terms.
