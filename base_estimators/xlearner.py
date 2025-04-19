import joblib
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.base import BaseEstimator, RegressorMixin, clone
from base_models import CustomDataset, NeuralNetwork

import logging 
logger = logging.getLogger(__name__)

        
class XLearner(BaseEstimator, RegressorMixin):
    def __init__(self, input_size, num_classes, treated_params, control_params, propensity_model, device, epochs=10):
        self.input_size = input_size
        self.num_classes = num_classes
        self.treated_params = treated_params
        self.control_params = control_params
        self.batch_size = treated_params['batch_size']
        self.epochs = epochs
        self.device = device
        self.propensity_model = propensity_model

        # === Model Architecture ===
        self.mu0 = NeuralNetwork(input_size, num_classes, control_params['hidden_size'], control_params['num_layers'], control_params['dropout_rate']).to(device)
        self.mu1 = NeuralNetwork(input_size, num_classes, treated_params['hidden_size'], treated_params['num_layers'], treated_params['dropout_rate']).to(device)
        self.tau0 = NeuralNetwork(input_size, 1, control_params['hidden_size'], control_params['num_layers'], control_params['dropout_rate']).to(device)
        self.tau1 = NeuralNetwork(input_size, 1, treated_params['hidden_size'], treated_params['num_layers'], treated_params['dropout_rate']).to(device)

        # === Optimizers ===
        self.optimizer_mu0 = optim.Adam(self.mu0.parameters(), lr=control_params['learning_rate'])
        self.optimizer_mu1 = optim.Adam(self.mu1.parameters(), lr=treated_params['learning_rate'])
        self.optimizer_tau0 = optim.Adam(self.tau0.parameters(), lr=control_params['learning_rate'])
        self.optimizer_tau1 = optim.Adam(self.tau1.parameters(), lr=treated_params['learning_rate'])

        # === Loss Functions ===
        self.criterion_classification = nn.CrossEntropyLoss()
        self.criterion_regression = nn.MSELoss()

        # === Logs ===
        self.model_train_losses = np.zeros((self.epochs, 2))
        self.model_train_accuracies = np.zeros((self.epochs, 2))
        self.tau_losses = np.zeros((self.epochs, 2))

        logging.info("Initialized XLearner on device: %s", device)

    def _train_model(self, model, dataloader, criterion, optimizer, task_type="classification"):
        """
        Trains a single model for classification or regression, with loss and accuracy tracking.

        Args:
            model (torch.nn.Module): The model to train
            dataloader (DataLoader): DataLoader returning (X, y, w)
            optimizer: Optimizer instance
            task_type (str): "classification" or "regression"

        Returns:
            Tuple of:
                - model_train_losses: ndarray (epochs, 1)
                - model_train_accuracies: ndarray (epochs, 1)
        """
        model.to(self.device)
        model.train()

        losses = torch.zeros((self.epochs, 1))
        metrics = np.zeros((self.epochs, 1))  # Accuracy (classification) or 1 - MSE (regression)

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            correct_or_squared_error = 0.0
            total = 0

            for batch_X, batch_y, _ in dataloader:
                batch_X = batch_X.float().to(self.device)
                if task_type == "classification":
                    batch_y = batch_y.long().to(self.device)
                elif task_type == "regression":
                    batch_y = batch_y.float().view(-1, 1).to(self.device)
                else:
                    raise ValueError("Invalid task_type. Choose 'classification' or 'regression'.")

                # Forward
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Accumulate loss
                epoch_loss += loss.item()

                # Accuracy or MSE metric
                if task_type == "classification":
                    _, predicted = torch.max(predictions, 1)
                    total += batch_y.size(0)
                    correct_or_squared_error += (predicted == batch_y).sum().item()
                else:  # regression
                    total += batch_y.size(0)
                    correct_or_squared_error += torch.sum((predictions - batch_y) ** 2).item()

            # Final metrics
            avg_loss = epoch_loss / len(dataloader)
            if task_type == "classification":
                metric = correct_or_squared_error / total  # Accuracy
            else:
                mse = correct_or_squared_error / total
                metric = mse

            losses[epoch, 0] = avg_loss
            metrics[epoch, 0] = metric

            logging.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}, {'Accuracy' if task_type == 'classification' else 'MSE'}: {metric:.4f}")

        return losses, metrics

    def _get_prob_of_true_class(self, model, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(model(X_tensor), dim=1).cpu().numpy()
        return probs[np.arange(len(y)), y.astype(int)]

    def fit(self, X, y, t_ind):
        # === Split into Control and Treated ===
        mask_c, mask_t = t_ind == 0, t_ind == 1
        X_c, y_c, w_c = X[mask_c], y[mask_c], t_ind[mask_c]
        X_t, y_t, w_t = X[mask_t], y[mask_t], t_ind[mask_t]

        # === Train Response Models (mu0 and mu1) ===
        logging.info("Training mu0...")
        c_loader = DataLoader(CustomDataset(X_c, y_c, w_c), batch_size=self.batch_size, shuffle=True, drop_last=True)
        mu0_losses, mu0_acc = self._train_model(self.mu0, c_loader, self.criterion_classification, self.optimizer_mu0)

        logging.info("Training mu1...")
        t_loader = DataLoader(CustomDataset(X_t, y_t, w_t), batch_size=self.batch_size, shuffle=True, drop_last=True)
        mu1_losses, mu1_acc = self._train_model(self.mu1, t_loader, self.criterion_classification, self.optimizer_mu1)

        self.model_train_losses[:, 0] = mu0_losses[:, 0]
        self.model_train_losses[:, 1] = mu1_losses[:, 0]
        self.model_train_accuracies[:, 0] = mu0_acc[:, 0]
        self.model_train_accuracies[:, 1] = mu1_acc[:, 0]

        # === Compute Imputed Treatment Effects ===
        logging.info("Computing imputed treatment effects...")
        te0 = self._get_prob_of_true_class(self.mu1, X_c, y_c) - y_c  # using mu1 on control
        te1 = y_t - self._get_prob_of_true_class(self.mu0, X_t, y_t)  # using mu0 on treated

        # === Train Tau Models ===
        logging.info("Training tau0...")
        tau0_loader = DataLoader(CustomDataset(X_c, te0, w_c), batch_size=self.batch_size, shuffle=True, drop_last=True)
        tau0_losses, _ = self._train_model(self.tau0, tau0_loader, self.criterion_regression, self.optimizer_tau0, task_type="regression")

        logging.info("Training tau1...")
        tau1_loader = DataLoader(CustomDataset(X_t, te1, w_t), batch_size=self.batch_size, shuffle=True, drop_last=True)
        tau1_losses, _ = self._train_model(self.tau1, tau1_loader, self.criterion_regression, self.optimizer_tau1, task_type="regression")

        self.tau_losses[:, 0] = tau0_losses[:, 0]
        self.tau_losses[:, 1] = tau1_losses[:, 0]

        return self
    
    def predict_propensity(self, X):
        """
        Predicts propensity scores (P(T=1|X)) using the trained propensity model.

        Args:
            X (np.ndarray): Feature matrix of shape (N, D).

        Returns:
            np.ndarray: Predicted propensity scores for the treated class (P(T=1|X)).
        """
        # Create a dummy dataset since only X is used
        dummy_y = [0] * len(X)
        dummy_t = [0] * len(X)
        dataset = CustomDataset(X, dummy_y, dummy_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)

        self.propensity_model.eval()
        all_probs = []

        with torch.no_grad():
            for X_batch, _, _ in loader:
                X_batch = X_batch.float().to(self.device)
                logits = self.propensity_model(X_batch)
                probs = torch.softmax(logits, dim=1)
                all_probs.extend(probs[:, 1].cpu().numpy())  # Get P(T=1|X)
        
        return np.array(all_probs)

    def predict_outcome(self, X, y, t_ind, treated_flag):
        """
        Predict outcomes for either the treated or control model.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): True outcome labels.
            t_ind (np.ndarray): Treatment indicator (not used in prediction, included for dataset compatibility).
            treated_flag (bool): Whether to use the treated model (m1) or control model (m0).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: y_true, y_pred, y_pred_proba
        """
        model = self.mu1 if treated_flag else self.mu0
        model.eval()

        dataset = CustomDataset(X, y, t_ind)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)

        y_true, y_pred, y_pred_proba = [], [], []

        with torch.no_grad():
            for X_batch, y_batch, _ in loader:
                X_batch = X_batch.float().to(self.device)
                y_batch = y_batch.long().to(self.device)

                logits = model(X_batch)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                y_true.extend(y_batch.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())
                y_pred_proba.extend(probs.cpu().numpy())

        return np.array(y_true), np.array(y_pred), np.array(y_pred_proba)

    def predict_effect(self, X):
        self.mu0.eval()
        self.mu1.eval()
        self.tau0.eval()
        self.tau1.eval()

        propensity_scores = self.predict_propensity(X)
        propensity_scores = np.clip(propensity_scores, 1e-5, 1 - 1e-5)  # Ensure numerical stability
        
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            # Step 1: Predict factual outcomes for both treatment groups
            logits_treated = self.mu1(X_tensor).cpu()  # mu1(X), shape (N, 4)
            logits_control = self.mu0(X_tensor).cpu()  # mu0(X), shape (N, 4)
            predictions_treated = torch.softmax(logits_treated, dim=1).numpy()  # Probabilities for treated
            predictions_control = torch.softmax(logits_control, dim=1).numpy()  # Probabilities for control

            # Step 2: Compute imputed counterfactual outcomes
            imputed_control = torch.softmax(self.mu0(X_tensor).cpu(), dim=1).numpy()  # mu0(X) as counterfactual for treated
            imputed_treated = torch.softmax(self.mu1(X_tensor).cpu(), dim=1).numpy()  # mu1(X) as counterfactual for control

            # Step 3: Compute pseudo-outcomes (Difference between actual and imputed counterfactuals)
            D1 = predictions_treated - imputed_control  # Pseudo-outcome for treated (shape: N, 4)
            D0 = imputed_treated - predictions_control  # Pseudo-outcome for control (shape: N, 4)

            # Step 4: Compute treatment effect per class (Average of pseudo-outcomes)
            treatment_effect_per_class = (D1 + D0) / 2  # Shape: (N, 4)

            # Step 5: Use meta-learner τ(X) to refine treatment effect estimates per class
            tau0_predictions = torch.softmax(self.tau0(X_tensor).cpu(), dim=1).numpy()  # τ₀(X), shape (N, 4)
            tau1_predictions = torch.softmax(self.tau1(X_tensor).cpu(), dim=1).numpy()  # τ₁(X), shape (N, 4)


            # Compute propensity-weighted treatment effect
            final_treatment_effect = propensity_scores[:, None] * treatment_effect_per_class + \
                                    (1 - propensity_scores)[:, None] * (tau1_predictions-tau0_predictions)  # Shape: (N, 4)
        
        return final_treatment_effect  # Shape: (N, 4)
    
    def save(self, path):
        os.makedirs(path, exist_ok=True)

        # Save model weights
        torch.save(self.mu0.state_dict(), os.path.join(path, "mu0.pth"))
        torch.save(self.mu1.state_dict(), os.path.join(path, "mu1.pth"))
        torch.save(self.tau0.state_dict(), os.path.join(path, "tau0.pth"))
        torch.save(self.tau1.state_dict(), os.path.join(path, "tau1.pth"))

        # Save config/state
        joblib.dump({
            'input_size': self.input_size,
            'num_classes': self.num_classes,
            'control_params': self.control_params,
            'treated_params': self.treated_params,
            'epochs': self.epochs,
            'model_train_losses': self.model_train_losses,
            'model_train_accuracies': self.model_train_accuracies,
            'tau_losses': self.tau_losses
        }, os.path.join(path, "xlearner_config.pkl"))

        logging.info(f"✅ XLearner saved to {path}")

    def load(self, path, device):
        # Load config
        state = joblib.load(os.path.join(path, "xlearner_config.pkl"))
        self.device = device

        # Restore attributes
        self.input_size = state['input_size']
        self.num_classes = state['num_classes']
        self.control_params = state['control_params']
        self.treated_params = state['treated_params']
        self.epochs = state['epochs']
        self.model_train_losses = state['model_train_losses']
        self.model_train_accuracies = state['model_train_accuracies']
        self.tau_losses = state['tau_losses']

        # Rebuild models
        self.mu0 = NeuralNetwork(self.input_size, self.num_classes, self.control_params['hidden_size'],
                                self.control_params['num_layers'], self.control_params['dropout_rate']).to(device)

        self.mu1 = NeuralNetwork(self.input_size, self.num_classes, self.treated_params['hidden_size'],
                                self.treated_params['num_layers'], self.treated_params['dropout_rate']).to(device)

        self.tau0 = NeuralNetwork(self.input_size, 1, self.control_params['hidden_size'],
                                self.control_params['num_layers'], self.control_params['dropout_rate']).to(device)

        self.tau1 = NeuralNetwork(self.input_size, 1, self.treated_params['hidden_size'],
                                self.treated_params['num_layers'], self.treated_params['dropout_rate']).to(device)

        # Load weights
        self.mu0.load_state_dict(torch.load(os.path.join(path, "mu0.pth"), map_location=device))
        self.mu1.load_state_dict(torch.load(os.path.join(path, "mu1.pth"), map_location=device))
        self.tau0.load_state_dict(torch.load(os.path.join(path, "tau0.pth"), map_location=device))
        self.tau1.load_state_dict(torch.load(os.path.join(path, "tau1.pth"), map_location=device))

        self.mu0.to(device)
        self.mu1.to(device)
        self.tau0.to(device)
        self.tau1.to(device)

        logging.info(f"XLearner loaded from {path} on device: {device}")
