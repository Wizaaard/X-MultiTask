import numpy as np
import os
import pickle
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, mean_squared_error, silhouette_score, davies_bouldin_score, calinski_harabasz_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from base_models import CustomDataset, NeuralNetwork, MultiTaskNet, DCNet
from base_estimators.slearner import SLearner
from base_estimators.tlearner import TLearner
from base_estimators.xlearner import XLearner
from base_estimators.xmultitask_learner import XMultiTaskLearner
from base_estimators.dcn_pd_learner import DCNPDLearner

import logging 
logger = logging.getLogger(__name__)

CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_SAMPLES = 1000

def setup_logger(log_path=None, level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)

def get_learner(learner_name, config, input_size, num_classes, device, propensity_model=None):
    if learner_name == "SLearner":
        return SLearner(input_size=input_size,
                        num_classes=num_classes,
                        params=config["params"],
                        device=device)

    elif learner_name == "TLearner":
        return TLearner(input_size=input_size,
                        num_classes=num_classes,
                        treated_params=config["treated_params"],
                        control_params=config["control_params"],
                        device=device)

    elif learner_name == "XLearner":
        return XLearner(input_size=input_size,
                        num_classes=num_classes,
                        treated_params=config["treated_params"],
                        control_params=config["control_params"],
                        propensity_model=propensity_model,
                        device=device)
    
    elif learner_name == "DCNPDLearner":
        return DCNPDLearner(input_size=input_size,
                        num_classes=num_classes,
                        params=config["params"],
                        propensity_model=propensity_model,
                        device=device)

    elif learner_name == "XMultiTaskLearner":
        return XMultiTaskLearner(input_size=input_size,
                                 num_classes=num_classes,
                                 params=config["params"],
                                 propensity_model=propensity_model,
                                 device=device)
    else:
        raise ValueError(f"Unsupported learner type: {learner_name}")

def plot_training_curves(output_dir, epochs, model_train_losses, model_train_accuracies, tau_train_losses=None):
    """
    Plots and saves training curves for model loss, accuracy, and optional tau losses.

    Each plot is saved independently as a PDF.

    Args:
        epochs (int): Number of training epochs
        model_train_losses (np.ndarray): Shape (epochs, num_models) Overall outcome model(s) training losses
        model_train_accuracies (np.ndarray): Shape (epochs, num_models) Overall outcome model(s) training accuracies
        tau_train_losses (np.ndarray): Shape (epochs, num_models) Overall treatment model(s) training losses
        output_dir (str): Directory to save the plot
        filename (str): Filename for the saved plot (PDF)
    """
    x = range(1, epochs + 1)

    # === Plot Model Loss ===
    plt.figure(figsize=(8, 5))
    for i in range(model_train_losses.shape[1]):
        label = "Training loss" if model_train_losses.shape[1] == 1 else f"mu_{i} Loss"
        plt.plot(x, model_train_losses[:, i], label=label)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Outcome Prediction Model Training Loss Curves")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    loss_path = os.path.join(output_dir, f"outcome_model_training_loss.pdf")
    plt.savefig(loss_path, format='pdf', bbox_inches='tight')
    plt.close()
    logging.info(f"✅ Loss curve saved to: {loss_path}")

    # === Plot Model Accuracy ===
    plt.figure(figsize=(8, 5))
    for i in range(model_train_accuracies.shape[1]):
        label = "Training loss" if model_train_losses.shape[1] == 1 else f"mu_{i} Accuracy"
        plt.plot(x, model_train_accuracies[:, i], label=label)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Outcome Prediction Model Training Accuracy Curves")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    acc_path = os.path.join(output_dir, f"outcome_model_training_accuracy.pdf")
    plt.savefig(acc_path, format='pdf', bbox_inches='tight')
    plt.close()
    logging.info(f"✅ Accuracy curve saved to: {acc_path}")

    # === Plot Tau Loss (if available) ===
    if tau_train_losses is not None:
        plt.figure(figsize=(8, 5))
        for i in range(tau_train_losses.shape[1]):
            plt.plot(x, tau_train_losses[:, i], label=f"tau_{i} Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Treatment Effect Model (Tau) Loss Curves")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        tau_path = os.path.join(output_dir, f"treatment_model_training_tau_loss.pdf")
        plt.savefig(tau_path, format='pdf', bbox_inches='tight')
        plt.close()
        logging.info(f"✅ Tau loss curve saved to: {tau_path}")
