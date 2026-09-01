# GMSTA-TESR-model
Code and data processing scripts for the TESR statistical model used to quantify the influence of ENSO on global mean surface temperature anomalies, accompanying EGU preprint No. XXXX.

## Repository contents

### Core model

- `enso_gmst_model.py`  
  Main implementation of the TESR statistical model used to estimate the ENSO-related contribution to GMSTA.

### Analysis and visualization

- `plot_episodes.py`  
  Scripts for plotting and visualizing ENSO episodes and model results.

- `palette.py`  
  Color palettes and plotting settings used to ensure consistent figure styling across the analyses while improving accessibility for people with color vision deficiencies.

- `extensions2027.py`  
  Analysis and extensions related to the projected 2026–2027 period and associated model results.

### Bias and sensitivity analysis

- `run_bias_sensitivity_real.py`  
  Scripts for the bias and sensitivity analysis of the model using the relevant observational and/or model data.

- `benchmark_calibration.py`  
  Scripts for benchmark calibration and comparison used to assess the statistical model performance.

## Reproducibility

The scripts in this repository are provided to support the reproducibility of the results presented in the accompanying EGU preprint.

Data used by the model are not necessarily redistributed with this repository. Please refer to the accompanying documentation and the data providers for information on data sources, access, and licensing.

## Citation

If you use this code, please cite the accompanying EGU preprint:

> [Full citation to be added]

## License

This project is released under the MIT License. See `LICENSE` for details.
