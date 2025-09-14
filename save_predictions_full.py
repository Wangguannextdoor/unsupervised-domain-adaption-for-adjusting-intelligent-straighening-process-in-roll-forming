# === save_predictions_full.py ===
"""
Predictions for 30 models × 7 datasets
- Load models and hyperparams from models_index.xlsx
- For source domain (DC04), test only on validation split:
  * If models_index.xlsx has Train_BlechNr/Val_BlechNr → filter by BlechNr
  * If models_index.xlsx has Train_Windows/Val_Windows → filter by window indices
- Save predictions to 210 Excel files (30×7)
- Save a metrics_summary.xlsx with MAE & R² for each model × dataset × criterion
"""

import os
import ast
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score

from run_optuna_window_block import (
    CNNBaselineModel, load_scaler_dict, standardize,
    FIXED_INPUT_COLS, POSITION_OUTPUT_MAP,
    SCALER_PATH, DEVICE
)

# =============================
# CONFIG AREA
# =============================
MODELS_INDEX_PATH = "model/models_index.xlsx"
MODEL_DIR = "model"
SAVE_DIR = "results/predictions_all"
# =============================

# Dataset list
TARGET_DOMAINS = [
    ("DC04_symmetrisch", "data/Load_Data_for_Modelling_Output_DC04_symmetrisch(training).xlsx"),
    ("DP_S", "data/DP_S.xlsx"),
    ("DC_AS", "data/DC_AS.xlsx"),
    ("DP_AS", "data/DP_AS.xlsx"),
    ("DC_AS_NORM", "data/DC_AS_NORM.xlsx"),
    ("DP_AS_NORM", "data/DP_AS_NORM.xlsx"),
    ("DP_S_NORM", "data/DP_S_NORM.xlsx")
]


def make_windows(X, y, window_size=10):
    X_out, y_out = [], []
    n = len(X)
    for i in range(0, n, window_size):
        if i + window_size <= n:
            X_out.append(X[i:i+window_size])
            y_out.append(y[i:i+window_size].mean(axis=0))
        else:
            X_last = X[i:]
            y_last = y[i:]
            if len(X_last) > 0:
                pad_len = window_size - len(X_last)
                X_padded = np.pad(X_last, ((0, pad_len), (0, 0)), mode="edge")
                y_mean = y_last.mean(axis=0)
                X_out.append(X_padded)
                y_out.append(y_mean)
    return np.array(X_out), np.array(y_out)


if __name__ == "__main__":

    os.makedirs(SAVE_DIR, exist_ok=True)
    # Load scaler parameters
    scaler_dict = load_scaler_dict(SCALER_PATH)

    models_index = pd.read_excel(MODELS_INDEX_PATH)

    metrics_records = []

    for _, row in models_index.iterrows():
        model_file = row["Model_File"]
        print("\n" + "="*60)
        print(f"[INFO] Processing model: {model_file}")

        # Process Input_Type
        input_type_raw = row["Input_Type"]
        if isinstance(input_type_raw, str):
            input_type = ast.literal_eval(input_type_raw)
        else:
            input_type = input_type_raw
        output_cols = [POSITION_OUTPUT_MAP[t] for t in input_type]
        print(f"[INFO] Output columns: {output_cols}")

        # Define CNN and load weights
        dummy_input_dim = len(FIXED_INPUT_COLS) * 10  # (features × window)
        model = CNNBaselineModel(
            input_dim=dummy_input_dim, output_dim=len(output_cols),
            batch_size=row["BATCH_SIZE"], lr=row["LR"],
            hidden_dim=row["HIDDEN_DIM"], dropout_rate=row["DROPOUT"],
            weight_decay=row["WEIGHT_DECAY"]
        )
        model_path = os.path.join(MODEL_DIR, model_file)
        model.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"[INFO] Model loaded from {model_path}")

        # Predict on each dataset
        for dname, dpath in TARGET_DOMAINS:
            print("-"*40)
            print(f"[INFO] Predicting dataset: {dname}")

            df = pd.read_excel(dpath)
            if df.empty:
                print(f"[WARNING] Dataset {dname} empty, skipping")
                continue

            # Standardization
            X_std = standardize(df, FIXED_INPUT_COLS, scaler_dict)
            y_std = standardize(df, output_cols, scaler_dict)

            # Make windows
            Xw, yw = make_windows(X_std, y_std, window_size=10)
            Xw = Xw.reshape(Xw.shape[0], 1, -1)
            #20% validation data test
            # Special handling for source domain
            if dname == "DC04_symmetrisch":
                if "Val_BlechNr" in row:
                    # Filter by BlechNr
                    val_blechs = str(row["Val_BlechNr"]).split(",")
                    df = df[df["BlechNr"].astype(str).isin(val_blechs)]
                    print(f"[INFO] Using Val_BlechNr split: {val_blechs}")
                    print(f"[INFO] Rows after filtering: {len(df)}")

                    # Recompute std + windows after filtering
                    X_std = standardize(df, FIXED_INPUT_COLS, scaler_dict)
                    y_std = standardize(df, output_cols, scaler_dict)
                    Xw, yw = make_windows(X_std, y_std, window_size=10)
                    Xw = Xw.reshape(Xw.shape[0], 1, -1)

                elif "Val_Windows" in row:
                    # Use pre-defined window indices
                    val_windows = [int(x) for x in str(row["Val_Windows"]).split(",") if x.strip().isdigit()]
                    print(f"[INFO] Using Val_Windows split: {val_windows[:10]} ... total={len(val_windows)}")
                    Xw, yw = Xw[val_windows], yw[val_windows]
            '''80% training data test
            if dname == "DC04_symmetrisch":
                if "Train_BlechNr" in row and pd.notna(row["Train_BlechNr"]):
                    # Filter by BlechNr (training set)
                    train_blechs = [b.strip() for b in str(row["Train_BlechNr"]).split(",") if b.strip()]
                    df["BlechNr"] = df["BlechNr"].astype(str)
                    df = df[df["BlechNr"].isin(train_blechs)]
                    # Sort according to train_blechs order
                    df["BlechNr"] = pd.Categorical(df["BlechNr"], categories=train_blechs, ordered=True)
                    df = df.sort_values(["BlechNr"]).reset_index(drop=True)

                    print(f"[INFO] Using Train_BlechNr split: {train_blechs}")
                    print("[DEBUG] Current df BlechNr order:", df["BlechNr"].unique()[:10])
                    print(f"[INFO] Rows after filtering: {len(df)}")

                    # Recompute std + windows after filtering
                    X_std = standardize(df, FIXED_INPUT_COLS, scaler_dict)
                    y_std = standardize(df, output_cols, scaler_dict)
                    Xw, yw = make_windows(X_std, y_std, window_size=10)
                    Xw = Xw.reshape(Xw.shape[0], 1, -1)

                elif "Train_Windows" in row and pd.notna(row["Train_Windows"]):
                    # Use pre-defined window indices (training set)
                    train_windows = [int(x) for x in str(row["Train_Windows"]).split(",") if x.strip().isdigit()]
                    print(f"[INFO] Using Train_Windows split: {train_windows[:10]} ... total={len(train_windows)}")
                    Xw, yw = Xw[train_windows], yw[train_windows]
            '''
            print(f"[INFO] Final windows to predict: {len(Xw)}")

            if len(Xw) == 0:
                print(f"[WARNING] No windows for dataset {dname}, skipping")
                continue

            # Prediction (standardized space)
            y_pred_std = model.predict(Xw)
            print(f"[INFO] Prediction done, shape={y_pred_std.shape}")

            # Inverse standardization
            y_pred_orig = np.zeros_like(y_pred_std)
            for j, col in enumerate(output_cols):
                mean, std = scaler_dict[col]["mean"], scaler_dict[col]["std"]
                y_pred_orig[:, j] = y_pred_std[:, j] * std + mean

            # Ground truth (original)
            y_true_orig = df[output_cols].values
            y_true_orig_win = []
            n = len(y_true_orig)
            for i in range(0, n, 10):
                if i + 10 <= n:
                    y_true_orig_win.append(y_true_orig[i:i+10].mean(axis=0))
                else:
                    y_last = y_true_orig[i:]
                    if len(y_last) > 0:
                        y_true_orig_win.append(y_last.mean(axis=0))
            y_true_orig_win = np.array(y_true_orig_win)

            # Align lengths
            min_len = min(len(y_true_orig_win), len(yw), len(y_pred_std))
            y_true_orig_win = y_true_orig_win[:min_len]
            yw = yw[:min_len]
            y_pred_std = y_pred_std[:min_len]
            y_pred_orig = y_pred_orig[:min_len]
            print(f"[INFO] Aligning to min_len={min_len}")

            # Save results + collect metrics
            results_df = pd.DataFrame()
            for j, col in enumerate(output_cols):
                results_df[f"{col}_true_orig"] = y_true_orig_win[:, j]
                results_df[f"{col}_true_std"] = yw[:, j]
                results_df[f"{col}_pred_std"] = y_pred_std[:, j]
                results_df[f"{col}_pred_orig"] = y_pred_orig[:, j]

                # Compute metrics
                mae = mean_absolute_error(y_true_orig_win[:, j], y_pred_orig[:, j])
                r2 = r2_score(y_true_orig_win[:, j], y_pred_orig[:, j])
                metrics_records.append({
                    "Model_File": model_file,
                    "criterion": input_type[j],
                    "test": dname,
                    "MAE": mae,
                    "R2": r2
                })
                print(f"[METRIC] {dname} - {col}: MAE={mae:.4f}, R2={r2:.4f}")

            save_name = f"{os.path.splitext(model_file)[0]}_{dname}.xlsx"
            save_path = os.path.join(SAVE_DIR, save_name)
            results_df.to_excel(save_path, index=False)
            print(f"[INFO] Saved predictions to {save_path}")

    # Save metrics summary
    df_metrics = pd.DataFrame(metrics_records)
    summary_path = os.path.join(SAVE_DIR, "metrics_summary.xlsx")
    df_metrics.to_excel(summary_path, index=False)
    print("\n" + "="*60)
    print(f"[OK] Saved metrics summary to {summary_path}")
