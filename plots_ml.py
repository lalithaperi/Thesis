import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve,
    precision_recall_curve, auc
)

# -------- Paths --------
BASE = "C:/Users/CGaddam/Documents/Thesis/log-anomaly-thesis"
FEATURE_DIR = os.path.join(BASE, "features")
MODEL_DIR = os.path.join(BASE, "models")
PLOTS_DIR = os.path.join(BASE, "plots", "all_models")
os.makedirs(PLOTS_DIR, exist_ok=True)

CONFIGS = ["win10_stride5", "win20_stride10", "win30_stride10", "win50_stride20"]

# -------- Utilities --------
def plot_confusion(y_true, y_pred, tag):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {tag}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"conf_matrix_{tag}.png"))
    plt.close()

def plot_roc_pr(y_true, scores, tag):
    fpr, tpr, _ = roc_curve(y_true, scores)
    precision, recall, _ = precision_recall_curve(y_true, scores)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.2f}")
    plt.title("ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"roc_pr_{tag}.png"))
    plt.close()

def find_best_threshold_pr(scores, y_true):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]

# -------- Main Loop --------
for config in CONFIGS:
    print(f"\n🔍 Processing: {config}")
    y_path = os.path.join(FEATURE_DIR, f"y_ml_{config}.npy")
    if not os.path.exists(y_path):
        print(f"❌ Missing labels for {config}")
        continue
    y = np.load(y_path)

    # ----- LSTM -----
    try:
        model = load_model(os.path.join(MODEL_DIR, f"lstm_model_{config}.h5"))
        X_lstm = np.load(os.path.join(FEATURE_DIR, f"X_lstm_{config}.npy"))
        X_mcv = np.load(os.path.join(FEATURE_DIR, f"X_mcv_{config}.npy"))
        y_probs = model.predict([X_lstm, X_mcv]).flatten()
        y_pred = (y_probs > 0.5).astype(int)
        print("[LSTM]")
        print(classification_report(y, y_pred))
        plot_confusion(y, y_pred, f"lstm_{config}")
        plot_roc_pr(y, y_probs, f"lstm_{config}")
    except Exception as e:
        print(f"⚠️ LSTM model skipped: {e}")

    # ----- Random Forest -----
    try:
        model = joblib.load(os.path.join(MODEL_DIR, f"rf_{config}.pkl"))
        X = np.load(os.path.join(FEATURE_DIR, f"X_ml_{config}.npy"))
        y_probs = model.predict_proba(X)[:, 1]
        y_pred = (y_probs > 0.5).astype(int)
        print("[Random Forest]")
        print(classification_report(y, y_pred))
        plot_confusion(y, y_pred, f"rf_{config}")
        plot_roc_pr(y, y_probs, f"rf_{config}")
    except Exception as e:
        print(f"⚠️ RF model skipped: {e}")

    # ----- SVM -----
    try:
        model = joblib.load(os.path.join(MODEL_DIR, f"svm_{config}.pkl"))
        X = np.load(os.path.join(FEATURE_DIR, f"X_ml_{config}.npy"))
        y_probs = model.predict_proba(X)[:, 1]
        y_pred = (y_probs > 0.5).astype(int)
        print("[SVM]")
        print(classification_report(y, y_pred))
        plot_confusion(y, y_pred, f"svm_{config}")
        plot_roc_pr(y, y_probs, f"svm_{config}")
    except Exception as e:
        print(f"⚠️ SVM model skipped: {e}")

    # ----- One-Class SVM -----
    try:
        model = joblib.load(os.path.join(MODEL_DIR, f"ocsvm_config_{config}.pkl"))
        X = np.load(os.path.join(FEATURE_DIR, f"X_ml_{config}.npy"))
        scores = -model.decision_function(X)
        threshold, _ = find_best_threshold_pr(scores, y)
        y_pred = (scores >= threshold).astype(int)
        print("[OCSVM]")
        print(classification_report(y, y_pred))
        plot_confusion(y, y_pred, f"ocsvm_{config}")
        plot_roc_pr(y, scores, f"ocsvm_{config}")
    except Exception as e:
        print(f"⚠️ OCSVM model skipped: {e}")

    # ----- Isolation Forest -----
    try:
        model = joblib.load(os.path.join(MODEL_DIR, f"if_config_{config}.pkl"))
        X = np.load(os.path.join(FEATURE_DIR, f"X_ml_{config}.npy"))
        scores = -model.decision_function(X)
        threshold, _ = find_best_threshold_pr(scores, y)
        y_pred = (scores >= threshold).astype(int)
        print("[Isolation Forest]")
        print(classification_report(y, y_pred))
        plot_confusion(y, y_pred, f"isoforest_{config}")
        plot_roc_pr(y, scores, f"isoforest_{config}")
    except Exception as e:
        print(f"⚠️ IF model skipped: {e}")

print("\n✅ All plots generated from saved models only — no retraining.")
