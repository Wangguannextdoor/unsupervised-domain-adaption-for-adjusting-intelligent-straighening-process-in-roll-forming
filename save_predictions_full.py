# === predict_all_models.py ===
"""
Batch prediction for 30 models × 7 datasets
- Models and hyperparameters are loaded from models_index.xlsx
- Source domain (DC04_symmetrisch): locate validation set using Val_Windows or Val_BlechNr
- Target domains: use full dataset directly
- Each prediction is saved to Excel (total 210 files)
- Compute MAE and R², and finally save metrics_summary.xlsx as summary
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

# Dataset list
TARGET_DOMAINS = [
    ("DC04_symmetrisch", "data/DC_S_TRAIN.xlsx"),
    ("DP_S", "data/DP_S.xlsx"),
    ("DC_AS", "data/DC_AS.xlsx"),
    ("DP_AS", "data/DP_AS.xlsx"),
    ("DC_AS_NORM", "data/DC_AS_NORM.xlsx"),
    ("DP_AS_NORM", "data/DP_AS_NORM.xlsx"),
    ("DP_S_NORM", "data/DP_S_NORM.xlsx")
]
SOURCE_SPLIT_MODE = "val"
MODELS_INDEX_PATH = "model/models_index.xlsx"
MODEL_DIR = "model"
SAVE_DIR = "results/predictions_all"

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


def parse_list_like(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    s = str(value).strip()
    if not s:
        return []
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
        return [parsed]
    except Exception:
        return [x.strip() for x in s.split(",") if x.strip()]


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load model index
    models_index = pd.read_excel(MODELS_INDEX_PATH)
    scaler_dict = load_scaler_dict(SCALER_PATH)

    metrics_records = []

    for _, row in models_index.iterrows():
        model_file = row["Model_File"]
        print("\n" + "="*70)
        print(f"[INFO] Processing model: {model_file}")

        # Input_Type & output columns
        input_type = parse_list_like(row["Input_Type"])
        output_cols = [POSITION_OUTPUT_MAP[t] for t in input_type]

        # Initialize model
        model = CNNBaselineModel(
            input_dim=len(FIXED_INPUT_COLS) * 10,
            output_dim=len(output_cols),
            batch_size=int(row["BATCH_SIZE"]),
            lr=float(row["LR"]),
            hidden_dim=int(row["HIDDEN_DIM"]),
            dropout_rate=float(row["DROPOUT"]),
            weight_decay=float(row["WEIGHT_DECAY"])
        )
        model_path = os.path.join(MODEL_DIR, model_file)
        model.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"[INFO] Model weights loaded: {model_path}")

        # Iterate over datasets
        for dname, dpath in TARGET_DOMAINS:
            print("-"*50)
            print(f"[INFO] Predicting dataset: {dname}")

            df = pd.read_excel(dpath)
            if df.empty:
                print(f"[WARNING] {dname} is empty, skipping")
                continue

            # Source domain processing
            if dname == "DC04_symmetrisch":
                if SOURCE_SPLIT_MODE == "val":
                    if "Val_Windows" in row and pd.notna(row["Val_Windows"]):
                        val_windows = [int(x) for x in parse_list_like(row["Val_Windows"]) if str(x).isdigit()]
                        print(f"[INFO] Using Val_Windows to locate validation set, total {len(val_windows)} windows")

                        X_std = standardize(df, FIXED_INPUT_COLS, scaler_dict)
                        y_std = standardize(df, output_cols, scaler_dict)
                        Xw_all, yw_all = make_windows(X_std, y_std, window_size=10)

                        keep = [w for w in val_windows if 0 <= w < len(Xw_all)]
                        Xw, yw = Xw_all[keep], yw_all[keep]

                        y_true_orig = df[output_cols].values
                        y_true_orig_win = [y_true_orig[i:i + 10].mean(axis=0)
                                           for i in range(0, len(y_true_orig), 10)
                                           if i + 10 <= len(y_true_orig)]
                        y_true_orig_win = np.array(y_true_orig_win)[keep]

                    elif "Val_BlechNr" in row and pd.notna(row["Val_BlechNr"]):
                        val_blechs = parse_list_like(row["Val_BlechNr"])
                        print(f"[INFO] Using Val_BlechNr to locate validation set, count={len(val_blechs)}")
                        df = df[df["BlechNr"].astype(str).isin([str(b) for b in val_blechs])]
                        print(f"[INFO] Rows after filtering={len(df)}")

                        X_std = standardize(df, FIXED_INPUT_COLS, scaler_dict)
                        y_std = standardize(df, output_cols, scaler_dict)
                        Xw, yw = make_windows(X_std, y_std, window_size=10)

                        y_true_orig = df[output_cols].values
                        y_true_orig_win = [y_true_orig[i:i + 10].mean(axis=0)
                                           for i in range(0, len(y_true_orig), 10)
                                           if i + 10 <= len(y_true_orig)]
                        y_true_orig_win = np.array(y_true_orig_win)

                elif SOURCE_SPLIT_MODE == "train":
                    if "Train_Windows" in row and pd.notna(row["Train_Windows"]):
                        train_windows = [int(x) for x in parse_list_like(row["Train_Windows"]) if str(x).isdigit()]
                        print(f"[INFO] Using Train_Windows to locate training set, total {len(train_windows)} windows")

                        X_std = standardize(df, FIXED_INPUT_COLS, scaler_dict)
                        y_std = standardize(df, output_cols, scaler_dict)
                        Xw_all, yw_all = make_windows(X_std, y_std, window_size=10)

                        keep = [w for w in train_windows if 0 <= w < len(Xw_all)]
                        Xw, yw = Xw_all[keep], yw_all[keep]

                        y_true_orig = df[output_cols].values
                        y_true_orig_win = [y_true_orig[i:i + 10].mean(axis=0)
                                           for i in range(0, len(y_true_orig), 10)
                                           if i + 10 <= len(y_true_orig)]
                        y_true_orig_win = np.array(y_true_orig_win)[keep]

                    elif "Train_BlechNr" in row and pd.notna(row["Train_BlechNr"]):
                        train_blechs = parse_list_like(row["Train_BlechNr"])
                        print(f"[INFO] Using Train_BlechNr to locate training set, count={len(train_blechs)}")
                        df = df[df["BlechNr"].astype(str).isin([str(b) for b in train_blechs])]
                        print(f"[INFO] Rows after filtering={len(df)}")

                        X_std = standardize(df, FIXED_INPUT_COLS, scaler_dict)
                        y_std = standardize(df, output_cols, scaler_dict)
                        Xw, yw = make_windows(X_std, y_std, window_size=10)

                        y_true_orig = df[output_cols].values
                        y_true_orig_win = [y_true_orig[i:i + 10].mean(axis=0)
                                           for i in range(0, len(y_true_orig), 10)
                                           if i + 10 <= len(y_true_orig)]
                        y_true_orig_win = np.array(y_true_orig_win)
            else:
                print(f"[INFO] {dname} target domain, using full dataset")

                # Standardize input/output
                X_std = standardize(df, FIXED_INPUT_COLS, scaler_dict)
                y_std = standardize(df, output_cols, scaler_dict)

                # Make windows (val_ratio=0 means no split, full data)
                Xw, yw = make_windows(X_std, y_std, window_size=10)
                Xw = Xw.reshape(Xw.shape[0], 1, -1)

                # Original ground truth window means
                y_true_orig = df[output_cols].values
                y_true_orig_win = [
                    y_true_orig[i:i + 10].mean(axis=0)
                    for i in range(0, len(y_true_orig), 10)
                    if i + 10 <= len(y_true_orig)
                ]
                y_true_orig_win = np.array(y_true_orig_win)

            # Align
            min_len = min(len(y_true_orig_win), len(yw), len(Xw))
            Xw, yw, y_true_orig_win = Xw[:min_len], yw[:min_len], y_true_orig_win[:min_len]
            Xw = Xw.reshape(Xw.shape[0], 1, -1)

            # Prediction
            y_pred_std = model.predict(Xw)
            y_pred_orig = np.zeros_like(y_pred_std)
            for j, col in enumerate(output_cols):
                mean, std = scaler_dict[col]["mean"], scaler_dict[col]["std"]
                y_pred_orig[:, j] = y_pred_std[:, j] * std + mean

            # Save prediction results
            results_df = pd.DataFrame()
            for j, col in enumerate(output_cols):
                results_df[f"{col}_true_orig"] = y_true_orig_win[:, j]
                results_df[f"{col}_true_std"]  = yw[:, j]
                results_df[f"{col}_pred_std"]  = y_pred_std[:, j]
                results_df[f"{col}_pred_orig"] = y_pred_orig[:, j]

                # Compute MAE & R²
                mae = mean_absolute_error(y_true_orig_win[:, j], y_pred_orig[:, j])
                r2 = r2_score(y_true_orig_win[:, j], y_pred_orig[:, j])
                metrics_records.append({
                    "criterion": input_type[j],
                    "test": dname,
                    "Model_File": model_file,
                    "MAE": mae,
                    "R2": r2
                })
                print(f"[METRIC] {dname} - {col}: MAE={mae:.4f}, R2={r2:.4f}")

            save_name = f"{os.path.splitext(model_file)[0]}_{dname}.xlsx"
            save_path = os.path.join(SAVE_DIR, save_name)
            results_df.to_excel(save_path, index=False)
            print(f"[INFO] Results saved: {save_path}")

    # Summary
    df_metrics = pd.DataFrame(metrics_records)
    summary_path = os.path.join(SAVE_DIR, "metrics_summary.xlsx")
    df_metrics.to_excel(summary_path, index=False)
    print("\n" + "="*70)
    print(f"[OK] Metrics summary saved: {summary_path}")
