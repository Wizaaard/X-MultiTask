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


class TLearner(BaseEstimator, RegressorMixin):
    def __init__(self, input_size, num_classes, treated_params, control_params, device, epochs=15):
        self.input_size = input_size
        self.num_classes = num_classes
        self.treated_params = treated_params
        self.control_params = control_params
        self.batch_size = treated_params['batch_size']
        self.epochs = epochs
        self.device = device

        # Initialize models
        self.model_control = NeuralNetwork(input_size, num_classes, control_params['hidden_size'],
                                           control_params['num_layers'], control_params['dropout_rate']).to(device)

        self.model_treated = NeuralNetwork(input_size, num_classes, treated_params['hidden_size'],
                                           treated_params['num_layers'], treated_params['dropout_rate']).to(device)

        # Optimizers & loss
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer_control = optim.Adam(self.model_control.parameters(), lr=control_params['learning_rate'])
        self.optimizer_treated = optim.Adam(self.model_treated.parameters(), lr=treated_params['learning_rate'])

        # Training history
        self.model_train_losses = np.zeros((epochs, 2))      # 0 = control, 1 = treated
        self.model_train_accuracies = np.zeros((epochs, 2))  # 0 = control, 1 = treated

        logging.info("Initialized T-Learner on device: %s", device)

    def _train_model(self, model, dataloader, optimizer, task_type="classification"):
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
        metrics = np.zeros((self.epochs, 1))  # Accuracy (classification) or MSE (regression)

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
                loss = self.criterion(predictions, batch_y)

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

    def fit(self, X, y, t_ind):
        # Split data
        X_c, y_c, w_c = X[t_ind == 0], y[t_ind == 0], t_ind[t_ind == 0]
        X_t, y_t, w_t = X[t_ind == 1], y[t_ind == 1], t_ind[t_ind == 1]

        # Dataloaders
        c_loader = DataLoader(CustomDataset(X_c, y_c, w_c), batch_size=self.batch_size, shuffle=True, drop_last=True)
        t_loader = DataLoader(CustomDataset(X_t, y_t, w_t), batch_size=self.batch_size, shuffle=True, drop_last=True)

        # Train control model
        logging.info("Training model on control group...")
        losses_c, accs_c = self._train_model(self.model_control, c_loader, self.optimizer_control)

        # Train treated model
        logging.info("Training model on treated group...")
        losses_t, accs_t = self._train_model(self.model_treated, t_loader, self.optimizer_treated)

        # Store training logs
        self.model_train_losses[:, 0] = losses_c[:, 0]
        self.model_train_losses[:, 1] = losses_t[:, 0]
        self.model_train_accuracies[:, 0] = accs_c[:, 0]
        self.model_train_accuracies[:, 1] = accs_t[:, 0]

        return self

    def predict_outcome(self, X, y, t_ind, treated_flag=False):
        model = self.model_treated if treated_flag else self.model_control
        model.eval()
        y_true = []
        y_pred = []
        y_pred_proba = []

        dataset = CustomDataset(X, y, t_ind)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Inference on the test set
        with torch.no_grad():
            for X_batch, y_batch, _ in loader:
                X_batch, y_batch = X_batch.float().to(self.device), y_batch.long().to(self.device)

                # Get model predictions
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                y_true.extend(y_batch.cpu().numpy().astype(np.int64))
                y_pred.extend(predicted.cpu().numpy())
                y_pred_proba.extend(torch.softmax(outputs, dim=1).cpu().numpy())

        return np.array(y_true), np.array(y_pred), np.array(y_pred_proba)
        
    def predict_effect(self, X):
        # Generate dummy targets and treatment indicators since we don’t use them for prediction
        dummy_y = np.zeros(len(X))  # Placeholder labels
        dummy_t = np.zeros(len(X))  # Placeholder treatment indicators

        # Get predicted probabilities for control group
        _, _, y_pred_proba_control = self.predict_outcome(X, dummy_y, dummy_t, treated_flag=False)

        # Get predicted probabilities for treated group
        _, _, y_pred_proba_treated = self.predict_outcome(X, dummy_y, dummy_t, treated_flag=True)

        # Treatment effect = Treated prediction - Control prediction (class-wise)
        treatment_effect_per_class = y_pred_proba_treated - y_pred_proba_control

        return treatment_effect_per_class
    
    def save(self, path):
        os.makedirs(path, exist_ok=True)

        # Save model weights
        torch.save(self.model_control.state_dict(), os.path.join(path, "model_control.pth"))
        torch.save(self.model_treated.state_dict(), os.path.join(path, "model_treated.pth"))

        # Save config/state
        joblib.dump({
            'input_size': self.input_size,
            'num_classes': self.num_classes,
            'control_params': self.control_params,
            'treated_params': self.treated_params,
            'epochs': self.epochs,
            'model_train_losses': self.model_train_losses,
            'model_train_accuracies': self.model_train_accuracies
        }, os.path.join(path, "tlearner_config.pkl"))

        logging.info(f"✅ T-Learner saved to {path}")

    def load(self, path, device):
        # Load config
        state = joblib.load(os.path.join(path, "tlearner_config.pkl"))
        self.device = device

        self.input_size = state['input_size']
        self.num_classes = state['num_classes']
        self.control_params = state['control_params']
        self.treated_params = state['treated_params']
        self.epochs = state['epochs']
        self.model_train_losses = state['model_train_losses']
        self.model_train_accuracies = state['model_train_accuracies']

        # Rebuild models
        self.model_control = NeuralNetwork(self.input_size, self.num_classes, self.control_params['hidden_size'],
                                        self.control_params['num_layers'], self.control_params['dropout_rate']).to(device)

        self.model_treated = NeuralNetwork(self.input_size, self.num_classes, self.treated_params['hidden_size'],
                                        self.treated_params['num_layers'], self.treated_params['dropout_rate']).to(device)

        # Load weights
        self.model_control.load_state_dict(torch.load(os.path.join(path, "model_control.pth"), map_location=device))
        self.model_treated.load_state_dict(torch.load(os.path.join(path, "model_treated.pth"), map_location=device))

        self.model_control.to(device)
        self.model_treated.to(device)

        logging.info(f"T-Learner loaded from {path} on device: {device}")