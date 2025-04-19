import argparse
import os
import logging
import pickle
import json
import numpy as np
import torch

from utils import setup_logger, load_config, load_pickle, get_learner, plot_training_curves
from base_models import CustomDataset, NeuralNetwork, MultiTaskNet, DCNet

def main(args):
    # === Setup logging before anything else ===
    # Ensure output directory exists before logging starts
    os.makedirs(args.output_dir, exist_ok=True)

    # Now safely create the logger
    log_file = os.path.join(args.output_dir, f"{args.learner}_training.log")
    logger = setup_logger(log_path=log_file)
    logger = logging.getLogger(__name__)

    # === Load Data and Ensure they are all np.array===
    logging.info(f"Loading data from: {args.data_dir}")
    X_train = load_pickle(os.path.join(args.data_dir, "X_train.pkl")).astype(np.float32)#.values
    y_train = load_pickle(os.path.join(args.data_dir, "y_train.pkl")).astype(np.float32)
    w_train = load_pickle(os.path.join(args.data_dir, "w_train.pkl")).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    # === Load Config ===
    config = load_config(args.config)
    learner_config = config[args.learner]
    propensity_config = config.get("PropensityModel", None)
    
    # === Initialize Propensity Model if needed ===
    propensity_model = None
    if args.learner in ["XLearner", "XMultiTaskLearner", "DCNPDLearner"]:
        propensity_model = NeuralNetwork(input_size=input_size,
                                            num_classes= len(np.unique(w_train)),
                                            hidden_size=propensity_config["hidden_size"],
                                            num_layers=propensity_config["num_layers"],
                                            dropout_rate=propensity_config["dropout_rate"]).to(device)
        # Optionally: train or load pre-trained propensity model
        if args.propensity_model_path:
            logging.info(f"Loading pretrained propensity model from: {args.propensity_model_path}")
            state_dict = torch.load(args.propensity_model_path, map_location=device)
            propensity_model.load_state_dict(state_dict)
            propensity_model.eval()
    
    # === Initialize Learner ===
    learner = get_learner(args.learner, learner_config, input_size, num_classes, device, propensity_model)

    # === Train Learner ===
    logging.info(f"Training {args.learner}...")
    learner.fit(X_train, y_train, w_train)

    # === Save Training Curves ===
    logging.info("Saving training curves...")
    if args.learner in ["XLearner", "XMultiTaskLearner"]:
        plot_training_curves(args.output_dir, learner.epochs, learner.model_train_losses, learner.model_train_accuracies, learner.tau_losses)
    else:
        plot_training_curves(args.output_dir, learner.epochs, learner.model_train_losses, learner.model_train_accuracies)

    # === Save Learner ===
    learner.save(f"{args.output_dir}/checkpoints_{args.learner}/")
    logging.info(f"✅ Training complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train causal learner with config and pickled data")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with X_train.pkl, y_train.pkl, w_train.pkl")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--learner", type=str, choices=["SLearner", "TLearner", "XLearner", "XMultiTaskLearner", "DCNPDLearner"], required=True)
    parser.add_argument("--propensity_model_path", type=str, default=None,
                    help="Path to a pretrained propensity model (optional)")
    parser.add_argument("--config", type=str, default="./configs/learners_config.json", help="Path to JSON config file (default: ./configs/learners_config.json)")

    args = parser.parse_args()

    main(args)