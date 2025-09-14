# unsupervised-domain-adaption-for-adjusting-intelligent-straighening-process-in-roll-forming

This repository implements a complete pipeline for modeling rolling process data using convolutional neural networks (CNNs) with window+block data splitting. 
The goal of this repository is to find out the performance of models trained by source domain data(DC04_S) to make predictions on target domains(6 in total)
There are 3 positional deviations to predict seprately:
- `X_opt - X_Ist`
- `Y_opt - Y_Ist`
- `phi_opt - phi_Ist`

---

## Workflow

The full workflow contains three main steps:

### 1. Hyperparameter Search (`run_optuna_window_block.py`)
- Load standardization parameters from `Standardization_params.xlsx`.
- Standardize input and output features.
- Run Optuna hyperparameter optimization for `x`, `y`, and `phi` models.
- Save the best parameter combinations to `results/optuna_window_block_best.xlsx`.

### 2. Model Training (`train_window_block_batch.py`)
- Read the best hyperparameters from `optuna_window_block_best.xlsx`.
- Perform 10 random splits on the source domain data (`BlechNr` or `Window` mode).
- Train **30 models** in total (`x`, `y`, `phi` × 10 repeats).
- Save models into the `model/` directory.
- Generate `model/models_index.xlsx`, which records:
  - Model file names
  - Input type
  - Hyperparameters
  - Data split information (`Train_BlechNr` / `Val_BlechNr` or `Train_Windows` / `Val_Windows`).

### 3. Prediction & Evaluation (`save_predictions_full.py`)
- Load `models_index.xlsx` and all trained models.
- Run predictions on **7 datasets** (source + 6 targets).
- In the **source domain (DC04)**:
  - Only use the validation split (`Val_BlechNr` or `Val_Windows`).
- In the **target domains**:
  - Use the full dataset for prediction.
- Save results to **210 Excel files** (30 models × 7 datasets).
- Collect metrics (MAE & R²) and save them into `results/predictions_all/metrics_summary.xlsx`.

---

## Directory Structure

