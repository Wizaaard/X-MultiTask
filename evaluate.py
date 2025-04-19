import os
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, classification_report
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample

import logging 
logger = logging.getLogger(__name__)

CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_SAMPLES = 1000

def bootstrap_auc(learner, X, y, w, treated_flag=False, num_bootstrap=BOOTSTRAP_SAMPLES, confidence=CONFIDENCE_LEVEL):
    """Compute AUC and confidence interval using bootstrapping.

    Args:
        learner: 
        X, y, w: Input data
        num_bootstrap: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95)

    Returns:
        auc_results: dict[class_index] = (mean, lower, upper)
    """
    bootstrapped_aucs = {i: [] for i in range(learner.num_classes)}
    n = len(y)

    for _ in range(num_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        X_resampled = X[indices]
        y_resampled = y[indices]
        t_resampled = w[indices]

        y_true, _, y_pred_proba = learner.predict_outcome(X_resampled, y_resampled, t_resampled, treated_flag)
        y_true_one_hot = np.eye(learner.num_classes)[y_true]

        for i in range(learner.num_classes):
            if len(np.unique(y_true_one_hot[:, i])) < 2:
                continue
            auc = roc_auc_score(y_true_one_hot[:, i], y_pred_proba[:, i])
            bootstrapped_aucs[i].append(auc)

    auc_results = {}
    for i in range(learner.num_classes):
        if bootstrapped_aucs[i]:
            auc_mean = np.mean(bootstrapped_aucs[i])
            lower = np.percentile(bootstrapped_aucs[i], (1 - confidence) / 2 * 100)
            upper = np.percentile(bootstrapped_aucs[i], (1 + confidence) / 2 * 100)
            auc_results[i] = (auc_mean, lower, upper)

    return auc_results

def plot_roc_curves(y_true, y_pred_proba, auc_results, num_classes, output_dir=None, filename="roc_curves.pdf"):
    plt.figure(figsize=(10, 8))
    y_true_one_hot = np.eye(num_classes)[y_true]

    average_auc = 0
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_one_hot[:, i], y_pred_proba[:, i])

        if i in auc_results:
            auc_mean, auc_lower, auc_upper = auc_results[i]
            ci_range = (auc_upper - auc_lower) / 2
            average_auc += auc_mean
            plt.plot(fpr, tpr, label=f"Class {i} (AUC = {auc_mean:.2f} ± {ci_range:.2f})")

    average_auc /= num_classes
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid()
    plt.text(0.6, 0.2, f"Average AUC: {average_auc:.2f}", fontsize=12, bbox=dict(facecolor='white', alpha=0.6))

    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        logging.info(f"ROC curves saved to: {save_path}")

    plt.close()

def evaluate_outcome(learner, X, y, w, output_dir, treated_flag=False,
                     num_bootstrap=1000, confidence=0.95):
    """
    Evaluate model predictions, log classification performance, and save results.

    Args:
        learner: A learner object with `.predict_outcome()` and `.num_classes`
        X, y, w: Test data
        output_dir (str): Directory to save logs/plots
        num_bootstrap (int): Number of resamples
        confidence (float): Confidence level for AUC CI

    Returns:
        auc_results: Dict of class-level AUC means and confidence intervals
    """
    os.makedirs(output_dir, exist_ok=True)

    # === Predictions
    y_true, y_pred, y_pred_proba = learner.predict_outcome(X, y, w, treated_flag)

    # === Classification Report
    report = classification_report(
        y_true, y_pred,
        target_names=[f"Class {i}" for i in range(learner.num_classes)],
        output_dict=True  # To save as JSON
    )
    report_path = os.path.join(output_dir, "classification_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    logging.info(f"Classification report saved to: {report_path}")

    # === AUC via bootstrapping
    auc_results = bootstrap_auc(
        learner,
        X, y, w,
        treated_flag=treated_flag,
        num_bootstrap=num_bootstrap,
        confidence=confidence
    )
    auc_path = os.path.join(output_dir, "auc_results.json")
    with open(auc_path, "w") as f:
        json.dump({f"Class {k}": {"mean": v[0], "lower": v[1], "upper": v[2]} for k, v in auc_results.items()}, f, indent=4)
    logging.info(f"AUC results saved to: {auc_path}")

    # === ROC Curve Plot
    plot_roc_curves(y_true, y_pred_proba, auc_results, learner.num_classes, output_dir)

    # return auc_results

def find_multiclass_nearest_neighbors(X, y, w, num_classes, k=1):
    """
    Estimate treatment effects via nearest neighbor matching for multiclass outcomes.

    Args:
        X (np.ndarray): Feature matrix (N, d)
        y (np.ndarray): Class labels (N,)
        w (np.ndarray): Treatment assignments (0 = control, 1 = treated) (N,)
        num_classes (int): Number of outcome classes

    Returns:
        np.ndarray: Estimated treatment effect matrix (N, num_classes)
    """
    n = len(y)

    # One-hot encode Y
    y = y.astype(int) 
    y_one_hot = np.zeros((n, num_classes))
    y_one_hot[np.arange(n), y] = 1

    # Split data by treatment
    X_treated, y_treated = X[w == 1], y_one_hot[w == 1]
    X_control, y_control = X[w == 0], y_one_hot[w == 0]

    # Nearest neighbors
    nn_treated = NearestNeighbors(n_neighbors=k).fit(X_treated)
    nn_control = NearestNeighbors(n_neighbors=k).fit(X_control)

    # Match treated -> control
    _, idx_c = nn_control.kneighbors(X_treated)
    matched_control = y_control[idx_c.flatten()]

    # Match control -> treated
    _, idx_t = nn_treated.kneighbors(X_control)
    matched_treated = y_treated[idx_t.flatten()]

    # Estimate individual treatment effects
    tau_nn = np.zeros((n, num_classes))
    tau_nn[w == 1] = y_one_hot[w == 1] - matched_control
    tau_nn[w == 0] = matched_treated - y_one_hot[w == 0]

    return tau_nn

def compute_ci(values, alpha=0.05):
    lower = np.percentile(values, 100 * alpha / 2)
    upper = np.percentile(values, 100 * (1 - alpha / 2))
    mean = np.mean(values)
    return {
        "mean": float(mean),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "ci_half_width": float((upper - lower) / 2)
    }

def format_metric_string(ci_dict, precision=5):
    mean = round(ci_dict["mean"], precision)
    # error = round(ci_dict["ci_half_width"], precision)
    error = ci_dict["ci_half_width"]
    return f"{mean} +/- {error}"

def evaluate_treatment_effects(
    learner, X, y, w, output_dir,
    n_bootstrap=1000, confidence=0.95, seed=42
):
    """
    Evaluate treatment effects using bootstrap-based CI estimation.

    Args:
        learner: Model with `.predict_effect(X)` and `.num_classes`.
        X (np.ndarray): Feature matrix (N, d).
        y (np.ndarray): Class labels (N,).
        w (np.ndarray): Treatment indicators (N,).
        output_dir (str): Directory to save results.
        n_bootstrap (int): Number of bootstrap samples.
        confidence (float): Confidence level (e.g., 0.95).
        seed (int): Random seed for reproducibility.

    Saves:
        JSON file with PEHE and ATE error (per class and overall) with CI.
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)

    n = len(X)
    C = learner.num_classes
    pehe_boot = []
    ate_err_boot = []

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        X_boot = X[idx]
        y_boot = y[idx]
        w_boot = w[idx]

        # Recompute "true" and predicted effects on bootstrap sample
        tau_true_boot = find_multiclass_nearest_neighbors(X_boot, y_boot, w_boot, C)
        tau_pred_boot = learner.predict_effect(X_boot)

        # Compute metrics
        pehe_boot.append(np.mean((tau_true_boot - tau_pred_boot) ** 2, axis=0))
        ate_err_boot.append(np.abs(np.mean(tau_true_boot, axis=0) - np.mean(tau_pred_boot, axis=0)))

    pehe_boot = np.stack(pehe_boot)         # shape: (n_bootstrap, num_classes)
    ate_err_boot = np.stack(ate_err_boot)   # shape: (n_bootstrap, num_classes)

    def summarize(metric_array, name_prefix):
        return {
            f"{name_prefix} {i}": format_metric_string(
                compute_ci(metric_array[:, i], alpha=1 - confidence)
            )
            for i in range(C)
        }

    metrics = {
        "PEHE per class": summarize(pehe_boot, "class"),
        "Overall PEHE": format_metric_string(
            compute_ci(np.mean(pehe_boot, axis=1), alpha=1 - confidence)
        ),
        "ATE Error per class": summarize(ate_err_boot, "class"),
        "Overall ATE Error": format_metric_string(
            compute_ci(np.mean(ate_err_boot, axis=1), alpha=1 - confidence)
        )
    }

    # Save JSON
    output_path = os.path.join(output_dir, "treatment_effect_evaluation.json")
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)

# def evaluate_treatment_effects(learner, X, y, w, output_dir, n_bootstrap=1000, random_state=42):
#     """
#     Evaluate predicted treatment effects against nearest-neighbor estimates and save formatted results with 95% CI.
#     """
#     os.makedirs(output_dir, exist_ok=True)
#     np.random.seed(random_state)

#     # Point estimates
#     tau_true = find_multiclass_nearest_neighbors(X, y, w, learner.num_classes)
#     tau_pred = learner.predict_effect(X)
#     pehe = np.mean((tau_true - tau_pred) ** 2, axis=0)
#     ate_true = np.mean(tau_true, axis=0)
#     ate_pred = np.mean(tau_pred, axis=0)
#     ate_error = np.abs(ate_true - ate_pred)

#     # Bootstrap CI estimates
#     pehe_boot = []
#     ate_err_boot = []
#     for _ in range(n_bootstrap):
#         idx = resample(np.arange(len(X)))
#         tau_t_boot = tau_true[idx]
#         tau_p_boot = tau_pred[idx]
#         pehe_boot.append(np.mean((tau_t_boot - tau_p_boot) ** 2, axis=0))
#         ate_true_boot = np.mean(tau_t_boot, axis=0)
#         ate_pred_boot = np.mean(tau_p_boot, axis=0)
#         ate_err_boot.append(np.abs(ate_true_boot - ate_pred_boot))

#     pehe_boot = np.stack(pehe_boot)
#     ate_err_boot = np.stack(ate_err_boot)

#     def summarize(metric_array):
#         return [compute_ci(metric_array[:, i]) for i in range(metric_array.shape[1])]

#     def format_summary(metric_array, name_prefix):
#         summary = summarize(metric_array)
#         return {
#             f"{name_prefix} {i}": format_metric_string(summary[i])
#             for i in range(len(summary))
#         }

#     metrics = {
#         "PEHE per class": format_summary(pehe_boot, "class"),
#         "Overall PEHE": format_metric_string(compute_ci(np.mean(pehe_boot, axis=1))),
#         "ATE Error per class": format_summary(ate_err_boot, "class"),
#         "Overall ATE Error": format_metric_string(compute_ci(np.mean(ate_err_boot, axis=1)))
#     }

#     # Save JSON
#     output_path = os.path.join(output_dir, "treatment_effect_evaluation.json")
#     with open(output_path, "w") as f:
#         json.dump(metrics, f, indent=4)


# def evaluate_treatment_effects(learner, X, y, w, output_dir):
#     """
#     Evaluate predicted treatment effects against nearest-neighbor estimates and save results to JSON.

#     Args:
#         learner: Model with .predict_effect(X) and .num_classes
#         X (np.ndarray): Feature matrix (N, d)
#         y (np.ndarray): Class labels (N,)
#         w (np.ndarray): Treatment indicators (N,)
#         output_dir (str): Directory to save JSON evaluation results

#     Returns:
#         None
#     """
#     os.makedirs(output_dir, exist_ok=True)

#     # Estimate "true" treatment effect via nearest neighbor
#     tau_true = find_multiclass_nearest_neighbors(X, y, w, learner.num_classes)

#     # Predict treatment effect
#     tau_pred = learner.predict_effect(X)

#     # Compute PEHE and ATE error per class
#     pehe_per_class = np.mean((tau_true - tau_pred) ** 2, axis=0).tolist()
#     ate_true = np.mean(tau_true, axis=0)
#     ate_pred = np.mean(tau_pred, axis=0)
#     ate_error_per_class = np.abs(ate_true - ate_pred).tolist()

#     metrics = {
#         "PEHE per class": pehe_per_class,
#         "Overall PEHE": float(np.mean(pehe_per_class)),
#         "ATE Error per class": ate_error_per_class,
#         "Overall ATE Error": float(np.mean(ate_error_per_class))
#     }

#     # Save to JSON
#     output_path = os.path.join(output_dir, "treatment_effect_evaluation.json")
#     with open(output_path, "w") as f:
#         json.dump(metrics, f, indent=4)
