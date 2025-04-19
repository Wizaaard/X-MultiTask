import argparse
import os
import logging
import pickle
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from base_models import CustomDataset, NeuralNetwork, MultiTaskNet, DCNet
from utils import setup_logger, load_config, load_pickle, get_learner
from evaluate import bootstrap_auc, plot_roc_curves, evaluate_outcome, evaluate_treatment_effects


def main(args):
    # === Setup logging before anything else ===
    # Ensure output directory exists before logging starts
    os.makedirs(args.output_dir, exist_ok=True)

    # Now safely create the logger
    log_file = os.path.join(args.output_dir, f"{args.learner}_testing.log")
    logger = setup_logger(log_path=log_file)

    # === Load Data and Ensure they are all np.array===
    logging.info(f"Loading data from: {args.data_dir}")
    X_test = load_pickle(os.path.join(args.data_dir, "X_test.pkl")).astype(np.float32)#.values
    y_test = load_pickle(os.path.join(args.data_dir, "y_test.pkl")).astype(np.float32)
    w_test = load_pickle(os.path.join(args.data_dir, "w_test.pkl")).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = X_test.shape[1]
    num_classes = len(np.unique(y_test))

    # === Load Config ===
    config = load_config(args.config)
    learner_config = config[args.learner]
    propensity_config = config.get("PropensityModel", None)
    
    # === Initialize Propensity Model if needed ===
    propensity_model = None
    if args.learner in ["XLearner", "XMultiTaskLearner", "DCNPDLearner"]:
        propensity_model = NeuralNetwork(input_size=input_size,
                                            num_classes= len(np.unique(w_test)),
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
    learner.load(args.checkpoint, device)

    # === Evaluate ===
    logging.info("Evaluating outcome prediction for all treatment groups...")
    evaluate_outcome(
        learner=learner,
        X=X_test[w_test==0],
        y=y_test[w_test==0],
        w=w_test[w_test==0],
        treated_flag=False,
        output_dir=args.output_dir + '/Control'
    )
    evaluate_outcome(
        learner=learner,
        X=X_test[w_test==1],
        y=y_test[w_test==1],
        w=w_test[w_test==1],
        treated_flag=True,
        output_dir=args.output_dir + '/Treated'
    )
    logging.info("Evaluating treatment effect prediction for all treatment groups...")
    evaluate_treatment_effects(
        learner=learner,
        X=X_test,
        y=y_test,
        w=w_test,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test causal learner with config, checkpoints, and pickled data")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint directory")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with X_test.pkl, y_test.pkl, w_test.pkl")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--learner", type=str, choices=["SLearner", "TLearner", "XLearner", "XMultiTaskLearner", "DCNPDLearner"], required=True)
    parser.add_argument("--propensity_model_path", type=str, default=None,
                    help="Path to a pretrained propensity model (optional)")
    parser.add_argument("--config", type=str, default="./configs/learners_config.json", help="Path to JSON config file (default: ./configs/learners_config.json)")

    args = parser.parse_args()
    main(args)