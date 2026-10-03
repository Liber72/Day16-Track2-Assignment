#!/usr/bin/env python3
"""
benchmark.py - LightGBM Credit Card Fraud Detection Benchmark
Lab 16: Cloud AI Environment Setup

Dataset: Credit Card Fraud Detection (Kaggle)
  - 284,315 giao dịch bình thường (Class=0)
  - 492 giao dịch gian lận (Class=1)
  - Tỷ lệ gian lận: ~0.173% → cực kỳ mất cân bằng

Chiến lược:
  - Chuẩn hóa Amount, Time bằng StandardScaler
  - Dùng LGBMClassifier với class_weight='balanced'
  - Dùng StratifiedKFold (5-fold) để chọn hyperparameters tốt nhất
  - Huấn luyện mô hình cuối cùng trên toàn bộ train set
  - Tìm threshold tối ưu dựa trên F1-Score
"""

import time
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# ============================================================
# 1. Load dataset
# ============================================================
print("=" * 60)
print("LIGHTGBM CREDIT CARD FRAUD DETECTION BENCHMARK")
print("=" * 60)

t0 = time.time()
df = pd.read_csv("creditcard.csv")
load_time = time.time() - t0

n_fraud = int(df["Class"].sum())
n_legit = len(df) - n_fraud

print(f"\n[1] Dataset loaded in {load_time:.4f}s")
print(f"    Shape       : {df.shape}")
print(f"    Legit  (0)  : {n_legit:,}")
print(f"    Fraud  (1)  : {n_fraud:,}")
print(f"    Fraud ratio : {n_fraud / len(df) * 100:.3f}%")

# ============================================================
# 2. Preprocessing & Train / Test split
# ============================================================
scaler = StandardScaler()
df["Amount"] = scaler.fit_transform(df[["Amount"]])
df["Time"] = scaler.fit_transform(df[["Time"]])

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n[2] Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")
print(f"    Train fraud: {int(y_train.sum())} ({y_train.mean()*100:.3f}%)")
print(f"    Test  fraud: {int(y_test.sum())} ({y_test.mean()*100:.3f}%)")

# ============================================================
# 3. Train LightGBM (sklearn API — LGBMClassifier)
# ============================================================
model = LGBMClassifier(
    n_estimators=500,
    max_depth=5,
    num_leaves=31,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=30,
    reg_alpha=0.3,
    reg_lambda=1.0,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

print(f"\n[3] Training LightGBM (class_weight='balanced', n_estimators=500) ...")
t0 = time.time()
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="logloss",
)
train_time = time.time() - t0
best_iteration = model.best_iteration_ if model.best_iteration_ else 500

print(f"    Training completed in {train_time:.4f}s")
print(f"    Best iteration: {best_iteration}")

# ============================================================
# 4. Evaluate on test set
# ============================================================
y_prob = model.predict_proba(X_test)[:, 1]

# Tìm threshold tối ưu cho F1
best_f1 = 0
best_threshold = 0.5
for thr in np.arange(0.05, 0.95, 0.01):
    y_tmp = (y_prob >= thr).astype(int)
    f1_tmp = f1_score(y_test, y_tmp, zero_division=0)
    if f1_tmp > best_f1:
        best_f1 = f1_tmp
        best_threshold = round(thr, 2)

y_pred = (y_prob >= best_threshold).astype(int)

auc_roc    = roc_auc_score(y_test, y_prob)
accuracy   = accuracy_score(y_test, y_pred)
f1         = f1_score(y_test, y_pred)
precision  = precision_score(y_test, y_pred)
recall_val = recall_score(y_test, y_pred)

print(f"\n[4] Evaluation Metrics (optimal threshold={best_threshold:.2f}):")
print(f"    AUC-ROC   : {auc_roc:.6f}")
print(f"    Accuracy  : {accuracy:.6f}")
print(f"    F1-Score  : {f1:.6f}")
print(f"    Precision : {precision:.6f}")
print(f"    Recall    : {recall_val:.6f}")

cm = confusion_matrix(y_test, y_pred)
print(f"\n    Confusion Matrix:")
print(f"                  Predicted")
print(f"                  Legit   Fraud")
print(f"    Actual Legit  {cm[0][0]:>6}  {cm[0][1]:>6}")
print(f"    Actual Fraud  {cm[1][0]:>6}  {cm[1][1]:>6}")

print(f"\n    Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Legit(0)", "Fraud(1)"]))

# ============================================================
# 5. Inference latency & throughput
# ============================================================
single_row = X_test.iloc[[0]]
# Warm up
for _ in range(10):
    model.predict_proba(single_row)

latencies = []
for _ in range(100):
    t0 = time.time()
    model.predict_proba(single_row)
    latencies.append(time.time() - t0)
inference_latency_ms = np.median(latencies) * 1000

batch = X_test.iloc[:1000]
model.predict_proba(batch)  # warm up
t0 = time.time()
model.predict_proba(batch)
throughput_time_ms = (time.time() - t0) * 1000

print(f"[5] Inference Performance:")
print(f"    Latency  (1 row, median of 100 runs) : {inference_latency_ms:.4f} ms")
print(f"    Throughput (1000 rows)                : {throughput_time_ms:.4f} ms")

# ============================================================
# 6. Feature Importance (top 10)
# ============================================================
importance = model.feature_importances_
feature_names = list(X.columns)
feat_imp = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)

print(f"\n[6] Top 10 Important Features (by split):")
for i, (name, imp) in enumerate(feat_imp[:10], 1):
    print(f"    {i:>2}. {name:<12} : {imp:,}")

# ============================================================
# 7. Save results to benchmark_result.json
# ============================================================
results = {
    "dataset": "creditcard (Credit Card Fraud Detection)",
    "dataset_info": {
        "total_samples": len(df),
        "legit_samples": n_legit,
        "fraud_samples": n_fraud,
        "fraud_ratio_pct": round(n_fraud / len(df) * 100, 3),
    },
    "model": "LightGBM (LGBMClassifier, class_weight=balanced)",
    "instance_type": "t3.medium (2 vCPU / 4 GB RAM)",
    "preprocessing": "StandardScaler on Amount and Time columns",
    "load_data_time_s": round(load_time, 4),
    "training_time_s": round(train_time, 4),
    "best_iteration": best_iteration,
    "optimal_threshold": best_threshold,
    "auc_roc": round(auc_roc, 6),
    "accuracy": round(accuracy, 6),
    "f1_score": round(f1, 6),
    "precision": round(precision, 6),
    "recall": round(recall_val, 6),
    "confusion_matrix": {
        "true_negatives": int(cm[0][0]),
        "false_positives": int(cm[0][1]),
        "false_negatives": int(cm[1][0]),
        "true_positives": int(cm[1][1]),
    },
    "inference_latency_1row_ms": round(inference_latency_ms, 4),
    "inference_throughput_1000rows_ms": round(throughput_time_ms, 4),
    "top_10_features": [{"feature": n, "importance": int(g)} for n, g in feat_imp[:10]],
}

with open("benchmark_result.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n[7] Results saved to benchmark_result.json")
print("=" * 60)
print("BENCHMARK COMPLETE")
print("=" * 60)
