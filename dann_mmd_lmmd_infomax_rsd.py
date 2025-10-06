import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import mean_absolute_error, r2_score
import math

from helpers import load_scaler_dict, make_windows
from model import FeatureExtractor, Regressor, DomainClassifier, grad_reverse

# =============================
# 固定设置
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

SOURCE_PATH_TRANSFORMED = "data/new/DC_S_TRAIN_transformed.xlsx"
TARGET_PATH_TRANSFORMED = "data/new/DC_AS_transformed.xlsx"
SOURCE_PATH_ORIGINAL = "data/new/DC_S_TRAIN.xlsx"
TARGET_PATH_ORIGINAL = "data/new/DC_AS.xlsx"
SCALER_PATH = "data/new/Standardization_params_new.xlsx"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trial_results = []
# =============================
# warmup
# =============================
def linear_ramp(epoch, start, length):
    if epoch < start:
        return 0.0
    return min(1.0, (epoch - start) / max(1, length))
# =============================
# mmd
# =============================
def gaussian_kernel(x, y, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    x: [n, d], y: [m, d]
    """
    n, d = x.size()
    m, d = y.size()
    total = torch.cat([x, y], dim=0)  # [n+m, d]
    total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), d)
    total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), d)
    L2_distance = ((total0 - total1) ** 2).sum(2)  # [n+m, n+m]

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / ((n + m) ** 2)
    bandwidth = torch.clamp(bandwidth, min=1e-6)  # 避免太小
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bw) for bw in bandwidth_list]
    return sum(kernel_val)  # [n+m, n+m]

def mmd_rbf(x, y, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n = x.size(0)
    m = y.size(0)
    kernels = gaussian_kernel(x, y, kernel_mul, kernel_num, fix_sigma)
    XX = kernels[:n, :n]
    YY = kernels[n:, n:]
    XY = kernels[:n, n:]
    YX = kernels[n:, :n]
    loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)
    return loss
# =============================
# lmmd
# =============================
def lmmd_loss_bin(f_s, y_s, f_t, y_t_pred, num_bins=5, custom_bins=None):
    """
    f_s: [ns, d] 源域特征
    y_s: [ns, 1] 源域真实标签
    f_t: [nt, d] 目标域特征
    y_t_pred: [nt, 1] 目标域预测值 (来自 R)
    num_bins: 分桶数量
    custom_bins: 自定义分桶边界 (torch.tensor)，可选
    """
    # 如果没给 custom_bins，就用均分
    if custom_bins is None:
        bins = torch.linspace(y_s.min(), y_s.max(), num_bins+1).to(y_s.device)
    else:
        bins = custom_bins.to(y_s.device)

    # bucketize: 每个样本属于哪个区间
    s_bins = torch.bucketize(y_s.squeeze(), bins) - 1
    t_bins = torch.bucketize(y_t_pred.squeeze(), bins) - 1

    total_loss = 0.0
    count = 0
    for c in range(len(bins)-1):
        f_s_c = f_s[s_bins == c]
        f_t_c = f_t[t_bins == c]
        if len(f_s_c) > 1 and len(f_t_c) > 1:
            total_loss += mmd_rbf(f_s_c, f_t_c)
            count += 1
    if count > 0:
        total_loss /= count
    return torch.tensor(total_loss, device=f_s.device) if isinstance(total_loss, float) else total_loss
# =============================
# Infomax loss
# =============================
def infomax_loss_regression(y_preds, num_bins=20):
    """
    回归版 InfoMax:
    - 条件熵 ~ 预测不确定性 (方差)
    - 边际熵 ~ 直方图熵
    y_preds: [N, 1] 目标域预测值
    """
    # 条件熵: 预测方差 (简单版本：直接用 batch 内的方差)
    cond_entropy = torch.var(y_preds, dim=0).mean()

    # 边际熵: 直方图熵
    hist = torch.histc(y_preds.squeeze(),
                       bins=num_bins,
                       min=y_preds.min().item(),
                       max=y_preds.max().item())
    p = hist / (hist.sum() + 1e-6)
    p = p[p > 0]  # 避免 log(0)
    marg_entropy = - torch.sum(p * torch.log(p))

    # InfoMax 总损失: H(ŷ|z) - H(ŷ)
    return cond_entropy - marg_entropy, cond_entropy.item(), marg_entropy.item()

# =============================
# InfoMax score（for earlystopping）
# =============================
def infomax_score(F, R, X_tgt, device="cpu"):
    F.eval(); R.eval()
    with torch.no_grad():
        X_tgt_tensor = torch.tensor(X_tgt, dtype=torch.float32).to(device)
        f_tgt = F(X_tgt_tensor)
        y_pred = R(f_tgt).cpu().numpy()

    var_per_dim = np.var(y_pred, axis=0)
    score_var = np.mean(var_per_dim)

    score_entropy = 0.0
    for j in range(y_pred.shape[1]):
        hist, _ = np.histogram(y_pred[:, j], bins=20, density=True)
        hist = hist[hist > 0]
        score_entropy += -np.sum(hist * np.log(hist))
    score_entropy /= y_pred.shape[1]

    return score_var + score_entropy

# =============================
# RSD loss
# =============================
def rsd_loss(f_s, f_t, subspace_dim=None):
    """
    f_s: [ns, d] 源域特征
    f_t: [nt, d] 目标域特征
    subspace_dim: 子空间维度 (默认=min(d, rank))
    """
    # 中心化
    f_s = f_s - f_s.mean(0, keepdim=True)
    f_t = f_t - f_t.mean(0, keepdim=True)

    # SVD 分解
    U_s, _, _ = torch.svd(f_s)
    U_t, _, _ = torch.svd(f_t)

    if subspace_dim is None:
        subspace_dim = min(U_s.size(1), U_t.size(1))

    U_s = U_s[:, :subspace_dim]
    U_t = U_t[:, :subspace_dim]

    # 投影矩阵
    P_s = U_s @ U_s.t()
    P_t = U_t @ U_t.t()

    # Frobenius 范数差
    diff = P_s - P_t
    loss = torch.norm(diff, p='fro') ** 2
    return loss

# =============================
# RSD loss
# =============================
def mc_dropout_score(F, R, X_tgt, n_samples=10, device="cpu"):
    """
    Monte Carlo Dropout 不确定性评估
    返回目标域预测的平均方差
    """
    F.train()  # 保持dropout开启
    R.train()
    X_tgt_tensor = torch.tensor(X_tgt, dtype=torch.float32).to(device)

    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            f_tgt = F(X_tgt_tensor)
            y_pred = R(f_tgt).cpu().numpy()
            preds.append(y_pred)

    preds = np.stack(preds, axis=0)  # [n_samples, N, d]
    var = np.var(preds, axis=0).mean()
    return float(var)


# =============================
# 随机初始化脚本
# =============================
def set_seed(seed):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =============================
# 训练循环
# =============================
def train_dann_basic(source_X, source_y,
                     target_X, target_y,
                     input_dim, output_cols,
                     num_epochs=30, lr=5e-4,
                     hidden_dim=128, dropout=0.2,
                     batch_size=32,
                     weight_domain_loss=0.015,
                     weight_mmd=0.001,
                     weight_lmmd=0.001,
                     weight_infomax=0.01,
                     weight_rsd=0.001,
                     seed=42,
                     early_stopping_delta=0.005,
                     early_patience=3,
                     weight_decay=0.0,
                     target_X_val=None, target_y_val_orig=None, scaler_dict=None):


    set_seed(seed)
    src_ds = TensorDataset(torch.tensor(source_X, dtype=torch.float32),
                           torch.tensor(source_y, dtype=torch.float32))
    tgt_ds = TensorDataset(torch.tensor(target_X, dtype=torch.float32),
                           torch.zeros(len(target_X), 1))
    src_loader = DataLoader(src_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    tgt_loader = DataLoader(tgt_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    F = FeatureExtractor(input_dim, hidden_dim, dropout).to(DEVICE)
    R = Regressor(hidden_dim, output_dim=len(output_cols)).to(DEVICE)
    D = DomainClassifier(hidden_dim).to(DEVICE)

    optimizer = optim.Adam(
        list(F.parameters()) + list(R.parameters()) + list(D.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )
    criterion_label = nn.L1Loss()
    criterion_domain = nn.BCELoss()

    best_label_loss = float("inf")
    wait = 0
    history = []

    best_epoch = None
    best_mae = None
    best_r2 = None
    best_domain_loss = None
    best_mmd_loss = None
    best_lmmd_loss = None
    best_infomax_loss = None


    for epoch in range(1, num_epochs + 1):



        p_dann = linear_ramp(epoch, start=5, length=10)
        p_mmd = linear_ramp(epoch, start=0, length=5)
        p_lmmd = linear_ramp(epoch, start=8, length=12)  # 伪标签更稳时再上
        p_ifmx = linear_ramp(epoch, start=5, length=10)

        w_dann_eff = p_dann * weight_domain_loss
        w_mmd_eff = p_mmd * weight_mmd
        w_lmmd_eff = p_lmmd * weight_lmmd
        w_infomax_eff = p_ifmx * weight_infomax

        pp = epoch / num_epochs
        lam_grl = 2.0 / (1.0 + math.exp(-10 * pp)) - 1.0

        F.train(); R.train(); D.train()
        iter_tgt = iter(tgt_loader)

        epoch_label_loss = 0.0
        epoch_domain_loss = 0.0
        epoch_mmd_loss = 0.0
        epoch_lmmd_loss = 0.0
        epoch_infomax_loss = 0.0
        epoch_condH = 0.0
        epoch_margH = 0.0
        epoch_rsd_loss = 0.0

        for x_src, y_src in src_loader:
            try:
                x_tgt, _ = next(iter_tgt)
            except StopIteration:
                iter_tgt = iter(tgt_loader)
                x_tgt, _ = next(iter_tgt)

            x_src, y_src = x_src.to(DEVICE), y_src.to(DEVICE)
            x_tgt = x_tgt.to(DEVICE)

            f_src = F(x_src)
            f_tgt = F(x_tgt)
            y_pred = R(f_src)

            feat_all = torch.cat([f_src, f_tgt], dim=0)
            domain_pred = D(grad_reverse(feat_all, lam_grl))
            domain_labels = torch.cat([
                torch.zeros(len(f_src), 1),
                torch.ones(len(f_tgt), 1)
            ]).to(DEVICE)

            label_loss = criterion_label(y_pred, y_src)
            domain_loss = criterion_domain(domain_pred, domain_labels)
            mmd_loss = mmd_rbf(f_src, f_tgt)

            y_tgt_pseudo  = R(f_tgt).detach()
            lmmd_loss = lmmd_loss_bin(f_src, y_src, f_tgt, y_tgt_pseudo , num_bins=5)

            y_tgt_pred = R(f_tgt)
            infomax_loss, cond_H, marg_H = infomax_loss_regression(y_tgt_pred, num_bins=20)

            rsd = rsd_loss(f_src, f_tgt, subspace_dim=min(f_src.size(1), 20))  # 比如取前20维
            w_rsd_eff = p_mmd * weight_rsd  # 跟 MMD 类似的warmup调节

            loss = (label_loss
                    + w_dann_eff * domain_loss
                    + w_mmd_eff * mmd_loss
                    + w_lmmd_eff * lmmd_loss
                    + w_infomax_eff * infomax_loss
                    + w_rsd_eff * rsd)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_label_loss += label_loss.item()
            epoch_domain_loss += domain_loss.item()
            epoch_mmd_loss += mmd_loss.item()
            epoch_lmmd_loss += lmmd_loss.item()
            epoch_infomax_loss += infomax_loss.item()
            epoch_condH += cond_H
            epoch_margH += marg_H
            epoch_rsd_loss += rsd.item()

        avg_label_loss = epoch_label_loss / len(src_loader)
        avg_domain_loss = epoch_domain_loss / len(src_loader)
        avg_mmd_loss = epoch_mmd_loss / len(src_loader)
        avg_lmmd_loss = epoch_lmmd_loss / len(src_loader)
        avg_infomax_loss = epoch_infomax_loss / len(src_loader)
        avg_condH = epoch_condH / len(src_loader)
        avg_margH = epoch_margH / len(src_loader)
        avg_rsd_loss = epoch_rsd_loss / len(src_loader)
        score_mc = mc_dropout_score(F, R, target_X, n_samples=5, device=DEVICE)


        print(f"[Epoch {epoch}/{num_epochs}] "
              f"Label={avg_label_loss:.4f}, mc={score_mc:.4f}, Domain={avg_domain_loss:.4f}, "
              f"MMD={avg_mmd_loss:.4f}, LMMD={avg_lmmd_loss:.4f}, "
              f"InfoMax={avg_infomax_loss:.4f}, RSD={avg_rsd_loss:.4f}, "
              f"CondH={avg_condH:.4f}, MargH={avg_margH:.4f}")

        print(f"weight*loss to main loss "
              f"Label={avg_label_loss:.4f}, Domain*w={avg_domain_loss * w_dann_eff:.4f}, "
         #    f"MMD*w={avg_mmd_loss * w_mmd_eff:.4f}", LMMD*w={avg_lmmd_loss * w_lmmd_eff:.4f}",
              f"MMD*w={avg_mmd_loss * w_mmd_eff:.4f}, "
              f"InfoMax*w={avg_infomax_loss * w_infomax_eff:.4f}, "
              f"RSD*w={avg_rsd_loss * w_rsd_eff:.4f}")

        mae_val, r2_val = None, None
        if target_X_val is not None and target_y_val_orig is not None and scaler_dict is not None:
            mae_val, r2_val = evaluate_on_target(F, R, target_X_val, target_y_val_orig, output_cols, scaler_dict, device=DEVICE)
        history.append((epoch, avg_label_loss, mae_val, r2_val, avg_domain_loss, avg_mmd_loss, avg_lmmd_loss, avg_infomax_loss))
        # === MC Dropout score (目标域不确定性) ===

        # === Early stopping 逻辑 ===
        if avg_label_loss < best_label_loss - early_stopping_delta:
            best_label_loss = avg_label_loss
            wait = 0

            # ⬇️ 在 patience 重置时记录 best
            best_epoch = epoch
            best_mae = mae_val
            best_r2 = r2_val
            best_domain_loss = avg_domain_loss
            best_mmd_loss = avg_mmd_loss
            best_lmmd_loss = avg_lmmd_loss
            best_infomax_loss = avg_infomax_loss

            print(f"[Epoch {epoch}] Patience reset → record as current best (MAE={mae_val:.4f}, R²={r2_val:.4f})")
        else:
            if len(history) > 0:
                if len(history) > 1:  # 确保有上一个 epoch 的值
                    prev_score_mc = history[-2][-1]  # 存在history里最后一列保存score_mc
                    if score_mc > prev_score_mc:  # 方差比上一轮大
                        wait += 1
                        print(f"[Epoch {epoch}] Patience ↑ (mc_var {score_mc:.4f} > prev {prev_score_mc:.4f})")
                    else:
                        wait = max(0, wait - 1)
                        print(f"[Epoch {epoch}] Patience ↓ (mc_var {score_mc:.4f} <= prev {prev_score_mc:.4f})")

            if wait >= early_patience:
                print(f"Early stopping at epoch {epoch} (label + MC dropout)")
                break

        print(f"[Epoch {epoch}] Patience count = {wait}/{early_patience}")

        # 结束后不再用 min(history)，直接返回最后一次 patience=0 的 best
    score_infomax = infomax_score(F, R, target_X, device=DEVICE)
    score_mc = mc_dropout_score(F, R, target_X, n_samples=10, device=DEVICE)

    print(f"[Final Eval @Best Epoch={best_epoch}] "
          f"Target MAE={best_mae:.4f}, R²={best_r2:.4f}, "
          f"InfoMax={score_infomax:.4f}, MC_var={score_mc:.6f}, "
          f"DomainLoss={best_domain_loss:.4f}")

    return {
        "MAE": best_mae,
        "R2": best_r2,
        "epoch": best_epoch,
        "infomax": score_infomax,
        "mc_var": score_mc,
        "domain_loss": best_domain_loss,
        "mmd_loss": best_mmd_loss,
        "lmmd_loss": best_lmmd_loss,
        "infomax_loss": best_infomax_loss
    }


# =============================
# 目标域评估
# =============================
def evaluate_on_target(F, R, X_val, y_val_orig, output_cols, scaler_dict, device="cpu"):
    F.eval(); R.eval()
    with torch.no_grad():
        f_val = F(torch.tensor(X_val, dtype=torch.float32).to(device))
        y_pred_std = R(f_val).cpu().numpy()

    y_pred_orig = np.zeros_like(y_pred_std)
    for j, col in enumerate(output_cols):
        mean, std = scaler_dict[col]["mean"], scaler_dict[col]["std"]
        y_pred_orig[:, j] = y_pred_std[:, j] * std + mean

    mae = mean_absolute_error(y_val_orig, y_pred_orig)
    r2 = r2_score(y_val_orig, y_pred_orig)
    print(f"[Eval] Target Validation MAE={mae:.4f}, R²={r2:.4f}")
    return mae, r2


# =============================
# Optuna objective
# =============================
def objective(trial, X_src, y_src, X_tgt, y_tgt, X_val, y_val_orig, output_cols, scaler_dict):
    '''
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [5, 10, 20])
    weight_decay = trial.suggest_float("weight_decay", 1e-3, 5e-2, log=True)
    '''
    weight_domain_loss = trial.suggest_float("weight_domain_loss", 1e-4, 0.1)
    weight_mmd_loss = trial.suggest_float("weight_mmd_loss", 1e-4, 0.1)
    # weight_lmmd_loss = trial.suggest_float("weight_lmmd_loss", 1e-4, 0.1)
    weight_infomax_loss = trial.suggest_float("weight_infomax_loss", 1e-4, 0.1)
    # weight_dann_loss = trial.suggest_float("weight_dann_loss", 1e-4, 0.1)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-3, 5e-2, log=True)


    metrics = train_dann_basic(
        source_X=X_src, source_y=y_src,
        target_X=X_tgt, target_y=y_tgt,
        input_dim=X_src.shape[2], output_cols=output_cols,
        num_epochs=50,lr=lr,
        hidden_dim=128, dropout=0.2,
        batch_size=50,
        weight_domain_loss=weight_domain_loss,
        weight_mmd=weight_mmd_loss,
        weight_lmmd=0.0,
        weight_infomax=weight_infomax_loss,
        weight_rsd=0,
        seed=42,
        weight_decay=weight_decay,
        target_X_val=X_val, target_y_val_orig=y_val_orig, scaler_dict=scaler_dict,
        early_patience=3
    )

    # final_score = metrics["infomax"] - 5 * abs(metrics["domain_loss"] - 0.693)
    final_score = -(metrics["MAE"] + metrics["mc_var"])
    metrics["final_score"] = final_score

    print(f"[Trial {trial.number}] final_score={final_score:.4f}, "
          f"infomax={metrics['infomax']:.4f}, domain_loss={metrics['domain_loss']:.4f}, infomax_loss={metrics['infomax_loss']:.4f}")
    trial_results.append({
        "trial": trial.number,
        "MAE": metrics["MAE"],
        "R2": metrics["R2"],
        "epoch": metrics["epoch"],
        "mc_var": metrics["mc_var"],
        "final_score": final_score
    })

    return final_score


# =============================
# Main: 数据准备 + Optuna搜索
# =============================
if __name__ == "__main__":
    df_src = pd.read_excel(SOURCE_PATH_TRANSFORMED)
    df_tgt = pd.read_excel(TARGET_PATH_TRANSFORMED)
    df_tgt_orig = pd.read_excel(TARGET_PATH_ORIGINAL)
    scaler_dict = load_scaler_dict(SCALER_PATH)

    window_size = 10
    X_src = make_windows(df_src[FIXED_INPUT_COLS].values, window_size)
    X_tgt = make_windows(df_tgt[FIXED_INPUT_COLS].values, window_size)
    X_src = X_src.reshape(X_src.shape[0], 1, -1)
    X_tgt = X_tgt.reshape(X_tgt.shape[0], 1, -1)

    results = []
    for task in ["phi"]:#["x", "y", "phi"]:
        output_cols = [POSITION_OUTPUT_MAP[task]]
        y_src = df_src[output_cols].values[window_size - 1::window_size]
        y_tgt = df_tgt[output_cols].values[window_size - 1::window_size]

        X_val = make_windows(df_tgt[FIXED_INPUT_COLS].values, window_size)
        X_val = X_val.reshape(X_val.shape[0], 1, -1)
        y_val_orig = df_tgt_orig[output_cols].values[window_size - 1::window_size]

        print(f"\n=== Optuna Search: Task={task.upper()} ===")
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, X_src, y_src, X_tgt, y_tgt, X_val, y_val_orig, output_cols, scaler_dict),
                       n_trials=20)
        df_trials = pd.DataFrame(trial_results)
        save_path = f"results/optuna_dann_mmd_all_{task}.xlsx"
        df_trials.to_excel(save_path, index=False)
        print(f"\nAll trials for {task} saved to {save_path}")

        # 清空，避免下一个 task 混在一起
        trial_results.clear()
        print(f"Best trial params for {task}: {study.best_trial.params}")
        results.append({"Task": task, **study.best_trial.params})

    df_out = pd.DataFrame(results)
    save_path = "results/optuna_dann_best.xlsx"
    df_out.to_excel(save_path, index=False)
    print(f"\nBest hyperparameters saved to {save_path}")
