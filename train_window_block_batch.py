# === train_window_block_batch.py ===
"""
Train and save 30 models (x, y, phi × 10 random splits)
based on the best hyperparameter combinations in optuna_window_block_best.xlsx
- Save model files to model/
- Generate models_index.xlsx to record all models and hyperparameters
- You can select 2 mode:
    mode = 1 : BlechNr split
    mode = 2 : Window split
"""

import os
import numpy as np
import pandas as pd
import torch
from run_optuna_window_block import (
    CNNBaselineModel, load_scaler_dict, standardize,
    split_train_val_window_by_blechnr,
    split_train_val_window,
    FIXED_INPUT_COLS, POSITION_OUTPUT_MAP,
    SOURCE_PATH, SCALER_PATH, DEVICE
)


if __name__ == "__main__":
    os.makedirs("model", exist_ok=True)
    mode = 1
    # Load best hyperparameters
    best_df = pd.read_excel("results/optuna_window_block_best_BlechNr.xlsx")
    scaler_dict = load_scaler_dict(SCALER_PATH)

    model_records = []

    for _, row in best_df.iterrows():
        # Input_Type: stored as "['x']", "['y']", "['phi']" or list
        input_type = eval(row["Input_Type"]) if isinstance(row["Input_Type"], str) else row["Input_Type"]
        output_cols = [POSITION_OUTPUT_MAP[t] for t in input_type]

        # Load source domain data and standardize
        df = pd.read_excel(SOURCE_PATH)
        X = standardize(df, FIXED_INPUT_COLS, scaler_dict)
        y = standardize(df, output_cols, scaler_dict)

        # Extract best hyperparameters
        best_params = row.to_dict()

        for repeat in range(10):
            # Split data (window + block)
            if mode == 1:
                X_train, y_train, X_val, y_val, train_blechs, val_blechs = split_train_val_window_by_blechnr(
                    X, y, df, window_size=10, val_ratio=0.2, seed=repeat
                )
                X_train = X_train.reshape(X_train.shape[0], 1, -1)
                X_val = X_val.reshape(X_val.shape[0], 1, -1)
            elif mode == 2:
                X_train, y_train, X_val, y_val, train_ids, val_ids = split_train_val_window(
                    X, y, window_size=10, val_ratio=0.2, seed=repeat
                )
                X_train = X_train.reshape(X_train.shape[0], 1, -1)
                X_val = X_val.reshape(X_val.shape[0], 1, -1)
            # Define model
            model = CNNBaselineModel(
                input_dim=X_train.shape[2], output_dim=y_train.shape[1],
                batch_size=best_params["BATCH_SIZE"], lr=best_params["LR"],
                hidden_dim=best_params["HIDDEN_DIM"], dropout_rate=best_params["DROPOUT"],
                weight_decay=best_params["WEIGHT_DECAY"]
            )

            # Train
            model.train(X_train, y_train, epochs=30, X_val=X_val, y_val=y_val, cfg=best_params)

            # Save model
            type_str = "".join(input_type)  # "x" / "y" / "phi"
            model_name = f"best_{type_str}_repeat{repeat}.pth"
            save_path = os.path.join("model", model_name)
            torch.save(model.model.state_dict(), save_path)
            print(f"Model saved: {save_path}")

            # Record model info
            record = {"Model_File": model_name, "Input_Type": type_str}
            record.update(best_params)
            if mode == 1:
                record["Train_BlechNr"] = ",".join(map(str, train_blechs))
                record["Val_BlechNr"] = ",".join(map(str, val_blechs))
            elif mode == 2:
                record["Train_Windows"] = ",".join(map(str, train_ids))
                record["Val_Windows"] = ",".join(map(str, val_ids))

            model_records.append(record)

    # Save index table
    df_records = pd.DataFrame(model_records)
    index_path = "model/models_index.xlsx"
    df_records.to_excel(index_path, index=False)
    print(f"\nModel index saved to {index_path}")
