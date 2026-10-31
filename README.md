# Region-wise stacking ensembles for estimating brain-age using structural MRI

Code accompanying the paper:

> Georgios Antonopoulos, Shammi More, Simon B. Eickhoff, Federico Raimondo, Kaustubh R. Patil, for the Alzheimer's Disease Neuroimaging Initiative.
> **Region-wise stacking ensembles for estimating brain-age using structural MRI.**
> *Computers in Biology and Medicine*, 2025, 111182.
> DOI: [10.1016/j.compbiomed.2025.111182](https://doi.org/10.1016/j.compbiomed.2025.111182)

![Graphical abstract](https://ars.els-cdn.com/content/image/1-s2.0-S0010482525015355-ga1_lrg.jpg)

## Highlights

- Brain-region models effectively fuse voxel-wise information.
- Two-level stacking ensembles improve age prediction performance.
- Regional age predictions provide novel insights into brain aging.
- High performance without compromising privacy facilitates real-world applicability.

## Overview

Instead of averaging gray matter volume (GMV) within each brain region, we train a
two-level stacking ensemble:

1. **Level 0 (region-wise):** one model per parcel (873 parcels) predicts age from the
   voxel-wise GMV values within that parcel.
2. **Level 1 (brain-wise):** a second model combines the regional age predictions into
   a final brain-age estimate.

Several data fusion scenarios (site-specific vs. pooled training at each level,
out-of-sample level-0 predictions on the application site) are compared against a
baseline that uses regional mean GMV.

## Data

Models were trained and evaluated on four datasets of healthy individuals:

- [IXI](https://brain-development.org/ixi-dataset/)
- [eNKI](http://fcon_1000.projects.nitrc.org/indi/enhanced/) (enhanced Nathan Kline Institute Rockland Sample)
- [CamCAN](https://www.cam-can.org/)
- 1000BRAINS (referred to as `1000Gehirne` in the code)

The [ADNI](https://adni.loni.usc.edu/) dataset was used for validation in a clinical population.

The data are not included in this repository. Access must be requested from the
respective data providers.

## Repository contents

### Model training (R)

| File | Description |
|---|---|
| `parcelwise_level0.R` | Trains a level-0 model (default: `glmnet`) for one parcel on one dataset or combination of datasets. |
| `brainwise_multisite.R` | Trains level-1 models across sites and predicts new data. |
| `predict_newsite.R` | Command-line entry point used by `submit_new_predictions.sh`: predicts a test dataset with models trained on the given datasets (comma-separated), via `brainwise_multisite`. |
| `run_brainwise_multisite.R` | Runs `brainwise_multisite` for all train/test combinations of the four datasets. |
| `util.R` | Shared helper functions (cross-validation, metrics: MAE, R², correlation, bias, Wasserstein distance). |

### Cluster job submission (HTCondor)

All analyses were run on an [HTCondor](https://htcondor.org/) cluster.

| File | Description |
|---|---|
| `parcelwise_level0_submit.sh` | Generates HTCondor jobs for level-0 models (every parcel × every dataset combination). |
| `parcelwise_level1_submit.sh` | Generates HTCondor jobs for level-1 models. |
| `submit_new_predictions.sh` | Generates HTCondor jobs for predictions on held-out datasets. |
| `all_level0.submit`, `all_level1.submit` | Generated submit files used for the paper. |

Example:

```bash
./parcelwise_level0_submit.sh 1 > level0.submit
condor_submit level0.submit
```

Paths in the scripts (e.g. `/data/project/avg_vs_pred_stack/...`) refer to the original
cluster setup and need to be adapted.

### Analysis and figures (Python)

| File | Description |
|---|---|
| `l0dred_rda_to_csv.py` | Converts level-0 predictions from R `.rda` files to CSV. |
| `plot_performance.py` | Performance plots for the different fusion scenarios. |
| `train_size.py` | Effect of the number of training datasets on performance. |
| `pair_plots_corr_age.py` | Correlations between regional predictions and age. |
| `corr_gmvl0pred_age_acrsub.py` | Region-wise correlation of GMV and level-0 predictions with age, saved as brain maps. |
| `predict_dataset.py` | Classifies the source dataset from regional features (site effects). |
| `data_stats.py` | Descriptive statistics of the datasets. |
| `custom_lib.py` | Helper to write region-wise values into a NIfTI image using the 873-parcel atlas. |

## Requirements

- **R** (4.1.0) with `caret`, `glmnet`, `tidyverse`, `gtools`, `pbapply`, `transport`
- **Python 3** with `pandas`, `numpy`, `scipy`, `scikit-learn`, `julearn`, `nibabel`,
  `matplotlib`, `seaborn`, `rpy2`

## Citation

If you use this code, please cite:

```bibtex
@article{antonopoulos2025stacking,
  title   = {Region-wise stacking ensembles for estimating brain-age using structural MRI},
  author  = {Antonopoulos, Georgios and More, Shammi and Eickhoff, Simon B. and Raimondo, Federico and Patil, Kaustubh R.},
  journal = {Computers in Biology and Medicine},
  pages   = {111182},
  year    = {2025},
  doi     = {10.1016/j.compbiomed.2025.111182}
}
```

## Acknowledgements

Data used in preparation of this article were obtained from the Alzheimer's Disease
Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu).
