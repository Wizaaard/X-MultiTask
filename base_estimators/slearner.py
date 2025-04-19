import joblib
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.base import BaseEstimator, RegressorMixin, clone
from base_models import CustomDataset, NeuralNetwork

import logging 
logger = logging.getLogger(__name__)


class SLearner(BaseEstimator, RegressorMixin):
    def __init__(self, input_size, num_classes, params, device, epochs=15):
        self.input_size = input_size + 1  # +1 for treatment indicator
        self.num_classes = num_classes
        self.params = params
        self.device = device
        self.epochs = epochs

        self.hidden_size = params['hidden_size']
        self.num_layers = params['num_layers']
        self.dropout_rate = params['dropout_rate']
        self.learning_rate = params['learning_rate']
        self.batch_size = params['batch_size']

        self.model = NeuralNetwork(self.input_size, self.num_classes, self.hidden_size,
                                   self.num_layers, self.dropout_rate).to(device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.model_train_losses = np.zeros((epochs, 1))
        self.model_train_accuracies = np.zeros((epochs, 1))

        logging.info("Initialized S-Learner on device: %s", device)

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
        # Concatenate treatment indicator
        X_t = np.column_stack((X, t_ind))
        dataset = CustomDataset(X_t, y, t_ind)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        self.model_train_losses, self.model_train_accuracies = self._train_model(
            model=self.model,
            dataloader=dataloader,
            optimizer=self.optimizer
        )
        return self

    def predict_outcome(self, X, y, t_ind, treated_flag=False):
        self.model.eval()
        y_true, y_pred, y_pred_proba = [], [], []

        X_combined = np.column_stack([X, t_ind])
        
        dataset = CustomDataset(X_combined, y, t_ind)
        test_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        # Inference on the test set
        with torch.no_grad():
            for X_batch, y_batch, _ in test_loader:
                X_batch, y_batch = X_batch.float().to(self.device), y_batch.long().to(self.device)

                # Get model predictions
                outputs = self.model(X_batch)
                _, predicted = torch.max(outputs, 1)
                y_true.extend(y_batch.cpu().numpy().astype(np.int64))
                y_pred.extend(predicted.cpu().numpy())
                y_pred_proba.extend(torch.softmax(outputs, dim=1).cpu().numpy())  # Shape (N,4)

        return np.array(y_true), np.array(y_pred), np.array(y_pred_proba)

    def predict_effect(self, X):
        """
        Estimate treatment effect per class: P(Y=class | T=1, X) - P(Y=class | T=0, X)
        """
        self.model.eval()

        N = X.shape[0]
        dummy_y = np.zeros(N)  # y is unused for prediction
        t_ones = np.ones(N)
        t_zeros = np.zeros(N)

        # Predict probabilities under treated and control scenarios
        _, _, pred_proba_treated = self.predict_outcome(X, dummy_y, t_ones)
        _, _, pred_proba_control = self.predict_outcome(X, dummy_y, t_zeros)

        # Treatment effect per class = P(Y=c | T=1, X) - P(Y=c | T=0, X)
        treatment_effect_per_class = pred_proba_treated - pred_proba_control  # shape: (N, num_classes)

        return treatment_effect_per_class

    def save(self, path):
        os.makedirs(path, exist_ok=True)

        # Save model weights
        torch.save(self.model.state_dict(), os.path.join(path, "slearner_model.pth"))

        # Save config/state
        joblib.dump({
            'input_size': self.input_size - 1,  # minus treatment indicator
            'num_classes': self.num_classes,
            'params': self.params,
            'epochs': self.epochs,
            'model_train_losses': self.model_train_losses,
            'model_train_accuracies': self.model_train_accuracies,
        }, os.path.join(path, "slearner_config.pkl"))

        logging.info(f"S-Learner saved to {path}")

    def load(self, path, device):
        # Load config
        state = joblib.load(os.path.join(path, "slearner_config.pkl"))
        self.device = device

        self.input_size = state['input_size'] + 1  # restore treatment indicator
        self.num_classes = state['num_classes']
        self.params = state['params']
        self.epochs = state['epochs']
        self.model_train_losses = state['model_train_losses']
        self.model_train_accuracies = state['model_train_accuracies']

        self.hidden_size = self.params['hidden_size']
        self.num_layers = self.params['num_layers']
        self.dropout_rate = self.params['dropout_rate']
        self.learning_rate = self.params['learning_rate']
        self.batch_size = self.params['batch_size']

        self.model = NeuralNetwork(self.input_size, self.num_classes, self.hidden_size,
                                self.num_layers, self.dropout_rate).to(device)

        self.model.load_state_dict(torch.load(os.path.join(path, "slearner_model.pth"), map_location=device))
        self.model.to(device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        logging.info(f"S-Learner loaded from {path} on device: {device}")