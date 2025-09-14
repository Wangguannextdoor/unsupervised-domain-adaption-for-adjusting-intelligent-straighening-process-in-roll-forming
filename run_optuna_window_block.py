"""
Window+block mode full experiment
1. Read Standardization_params.xlsx and apply standardization by column names
2. Perform hyperparameter search for INPUT_TYPE=["x"], ["y"], ["phi"]
3. Keep only the best combination and save to optuna_window_block_best.xlsx
4. Train with 10 random BlechNr splits using the best combination
5. Test each model on source domain + 6 target domains
6. Save the final results to final_window_block_results.xlsx
- You can select 2 mode:
    mode = 1 : BlechNr split
    mode = 2 : Window split
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, r2_score
import optuna

# =============================
# Fixed input/output definitions
# =============================
FORCE_COLS = [
    '1-OW-OS Lateral Force', '1-OW-OS Axial Force',
    '2-OW-RS Lateral Force', '2-OW-RS Axial Force',
    '3-UW-RS Lateral Force', '3-UW-RS Axial Force',
    '4-UW-OS Lateral Force', '4-UW-OS Axial Force',
]
FIXED_INPUT_COLS = FORCE_COLS + ["X-Ist", "Y-Ist", "phi-Ist"]

POSITION_OUTPUT_MAP = {
    "x": "X_opt-X-Ist",
    "y": "Y_Opt-Y_ist",
    "phi": "phi_Opt-phi_ist",
}

SOURCE_PATH = "data/Load_Data_for_Modelling_Output_DC04_symmetrisch(training).xlsx"
SCALER_PATH = "data/Standardization_params.xlsx"
TARGET_DOMAINS = [
    SOURCE_PATH,
    "data/DP_S.xlsx",
    "data/DC_AS.xlsx",
    "data/DP_AS.xlsx",
    "data/DC_AS_NORM.xlsx",
    "data/DP_AS_NORM.xlsx",
    "data/DP_S_NORM.xlsx"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# =============================
# Model
# =============================
class CNNBaselineModel:
    def __init__(self, input_dim, output_dim,
                 batch_size=64, lr=0.001,
                 hidden_dim=128, dropout_rate=0.2,
                 weight_decay=0.0):
        self.batch_size = batch_size
        self.lr = lr
        self.model = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        ).to(DEVICE)

        self.criterion = nn.L1Loss(reduction="mean")
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=False
        )

    def train(self, X_train, y_train, epochs=30, X_val=None, y_val=None, cfg=None):
        dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                torch.tensor(y_train, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_loss = float("inf")
        patience_counter = 0
        clip_norm = cfg.get("CLIP_GRAD_NORM", None)

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                if clip_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_norm)
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(dataloader)

            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    Xv, yv = torch.tensor(X_val, dtype=torch.float32).to(DEVICE), torch.tensor(y_val, dtype=torch.float32).to(DEVICE)
                    yp = self.model(Xv)
                    val_loss = self.criterion(yp, yv).item()
                self.scheduler.step(val_loss)

                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if cfg.get("EARLY_STOPPING_PATIENCE") and patience_counter >= cfg["EARLY_STOPPING_PATIENCE"]:
                        break

    def predict(self, X):
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        return self.model(X).detach().cpu().numpy()


# =============================
# Standardization functions (from Excel parameters)
# =============================
def load_scaler_dict(scaler_path):
    df = pd.read_excel(scaler_path, sheet_name="Sheet1")
    scaler_dict = {row["column"]: {"mean": row["mean"], "std": row["scale"]}
                   for _, row in df.iterrows()}
    return scaler_dict

def standardize(df, cols, scaler_dict):
    arr = df[cols].copy()
    for c in cols:
        mean, std = scaler_dict[c]["mean"], scaler_dict[c]["std"]
        arr[c] = (arr[c] - mean) / std
    return arr.values


# =============================
# Data splitting: BlechNr + Window
# =============================
def split_train_val_window_by_blechnr(X, y, df, window_size=10, val_ratio=0.2, seed=42):
    blechnrs = df["BlechNr"].unique()
    np.random.seed(seed)
    val_blechnrs = np.random.choice(blechnrs, size=int(len(blechnrs)*val_ratio), replace=False)
    train_blechnrs = [b for b in blechnrs if b not in val_blechnrs]

    def make_windows(blechnr_list):
        X_out, y_out = [], []
        for b in blechnr_list:
            df_b = df[df["BlechNr"] == b]
            X_b, y_b = X[df["BlechNr"] == b], y[df["BlechNr"] == b]
            n = len(df_b)
            for i in range(0, n, window_size):
                if i + window_size <= n:
                    X_out.append(X_b[i:i+window_size])
                    y_out.append(y_b[i:i+window_size].mean(axis=0))
        return np.array(X_out), np.array(y_out)

    X_train, y_train = make_windows(train_blechnrs)
    X_val, y_val = make_windows(val_blechnrs)

    return X_train, y_train, X_val, y_val, train_blechnrs, val_blechnrs



def split_train_val_window(X, y, window_size=10, val_ratio=0.2, seed=42):

    n = len(X)
    print(f"[DEBUG] Original data: X={X.shape}, y={y.shape}, total rows={n}")

    X_out, y_out = [], []
    indices = []

    # Make windows
    for i in range(0, n, window_size):
        if i + window_size <= n:
            X_window = X[i:i+window_size]
            y_window = y[i:i+window_size].mean(axis=0)
            X_out.append(X_window)
            y_out.append(y_window)
            indices.append((i, i+window_size))
            print(f"[DEBUG] Window {i//window_size}: X_window={X_window.shape}, y_window={y_window.shape}")

    X_out = np.array(X_out)
    y_out = np.array(y_out)
    print(f"[DEBUG] After windowing: X_out={X_out.shape}, y_out={y_out.shape}")

    # Shuffle and split
    np.random.seed(seed)
    total = len(X_out)
    idx = np.arange(total)
    np.random.shuffle(idx)
    split = int(total * (1 - val_ratio))
    train_idx, val_idx = idx[:split], idx[split:]

    X_train, y_train = X_out[train_idx], y_out[train_idx]
    X_val, y_val = X_out[val_idx], y_out[val_idx]

    print(f"[DEBUG] Train set: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"[DEBUG] Validation set: X_val={X_val.shape}, y_val={y_val.shape}")
    print(f"[DEBUG] Train windows={len(train_idx)}, Validation windows={len(val_idx)}")

    return X_train, y_train, X_val, y_val, train_idx, val_idx


# =============================
# Evaluation
# =============================
def evaluate_model(model, X, y, output_cols):
    y_pred = model.predict(X)
    metrics = {"MAE": mean_absolute_error(y, y_pred),
               "R2": r2_score(y, y_pred),
               "per_dim": []}
    for i, col in enumerate(output_cols):
        metrics["per_dim"].append({
            "name": col,
            "MAE": mean_absolute_error(y[:, i], y_pred[:, i]),
            "R2": r2_score(y[:, i], y_pred[:, i])
        })
    return metrics


# =============================
# Optuna search space
# =============================
search_space_config = {
    "EPOCHS": [30],
    "LR": [1e-3, 5e-4],
    "HIDDEN_DIM": [64, 128, 256],
    "DROPOUT": [0.1, 0.3, 0.5],
    "WEIGHT_DECAY": [1e-4, 1e-5, 1e-6],
    "CLIP_GRAD_NORM": [None, 1.0, 5.0],
    "EARLY_STOPPING_PATIENCE": [3, 5, 10],
    "LOSS_WEIGHTS": [[1,1,1]],
    "BATCH_SIZE": [4, 8, 16]
}


def run_optuna_search(input_type, output_cols, X, y, df, mode=2):
    def objective(trial):
        cfg = {}
        cfg["EPOCHS"] = 30
        cfg["LR"] = trial.suggest_float("LR", min(search_space_config["LR"]), max(search_space_config["LR"]), log=True)
        cfg["HIDDEN_DIM"] = trial.suggest_categorical("HIDDEN_DIM", search_space_config["HIDDEN_DIM"])
        cfg["DROPOUT"] = trial.suggest_categorical("DROPOUT", search_space_config["DROPOUT"])
        cfg["WEIGHT_DECAY"] = trial.suggest_categorical("WEIGHT_DECAY", search_space_config["WEIGHT_DECAY"])
        cfg["CLIP_GRAD_NORM"] = trial.suggest_categorical("CLIP_GRAD_NORM", search_space_config["CLIP_GRAD_NORM"])
        cfg["EARLY_STOPPING_PATIENCE"] = trial.suggest_categorical("EARLY_STOPPING_PATIENCE", search_space_config["EARLY_STOPPING_PATIENCE"])
        cfg["LOSS_WEIGHTS"] = [1,1,1]
        cfg["BATCH_SIZE"] = trial.suggest_categorical("BATCH_SIZE", search_space_config["BATCH_SIZE"])

        # Choose splitting method
        if mode == 1:
            X_train, y_train, X_val, y_val, _, _ = split_train_val_window_by_blechnr(
                X, y, df, window_size=10, val_ratio=0.2, seed=42
            )
        elif mode ==2:
            X_train, y_train, X_val, y_val, _, _ = split_train_val_window(
                X, y, window_size=10, val_ratio=0.2, seed=42
            )

        X_train = X_train.reshape(X_train.shape[0], 1, -1)
        X_val = X_val.reshape(X_val.shape[0], 1, -1)

        model = CNNBaselineModel(X_train.shape[2], y_train.shape[1],
            batch_size=cfg["BATCH_SIZE"], lr=cfg["LR"],
            hidden_dim=cfg["HIDDEN_DIM"], dropout_rate=cfg["DROPOUT"],
            weight_decay=cfg["WEIGHT_DECAY"])
        model.train(X_train, y_train, epochs=30, X_val=X_val, y_val=y_val, cfg=cfg)
        metrics = evaluate_model(model, X_val, y_val, output_cols)
        return metrics["MAE"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=40)
    return study.best_trial.params


# =============================
# Main process
# =============================
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    scaler_dict = load_scaler_dict(SCALER_PATH)
    results = []

    for input_type in [["x"], ["y"], ["phi"]]:
        output_cols = [POSITION_OUTPUT_MAP[t] for t in input_type]
        df = pd.read_excel(SOURCE_PATH)
        X = standardize(df, FIXED_INPUT_COLS, scaler_dict)
        y = standardize(df, output_cols, scaler_dict)

        best_params = run_optuna_search(input_type, output_cols, X, y, df)
        row = {"Input_Type": input_type, **best_params}
        results.append(row)

    df_out = pd.DataFrame(results)
    save_path = "results/optuna_window_block_best.xlsx"
    df_out.to_excel(save_path, index=False)
    print(f"\nBest hyperparameters saved to {save_path}")
