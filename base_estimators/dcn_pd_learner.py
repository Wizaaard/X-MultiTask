import joblib
import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.base import BaseEstimator, RegressorMixin, clone
from base_models import CustomDataset, DCNet

import logging
logger = logging.getLogger(__name__)

# Deep Counterfactual Networks with Propensity-Dropout (DCNPD) was introduced in the paper https://arxiv.org/pdf/1706.05966
# Original code: https://github.com/shantanu48114860/Deep-Counterfactual-Networks-with-Propensity-Dropout

def shannon_entropy(p, eps=1e-8):
    p = p.clamp(eps, 1 - eps)
    return -p * torch.log(p) - (1 - p) * torch.log(1 - p)

def dropout_probability(entropy, gamma=1.0):
    return 1.0 - gamma / 2.0 - 0.5 * entropy


class DCNPDLearner(BaseEstimator, RegressorMixin):
    def __init__(self, input_size, num_classes, params, propensity_model, device, epochs=15):
        self.input_size = input_size
        self.num_classes = num_classes
        self.params = params
        self.learning_rate = params['learning_rate']
        self.batch_size = params['batch_size']
        self.num_tasks = params['num_tasks']
        self.epochs = epochs
        self.device = device
        self.propensity_model = propensity_model

        hidden_size = params['hidden_size']
        shared_layers = params['num_shared_layers']
        task_layers = params['num_task_specific_layers']

        self.mu_net = DCNet(input_size, hidden_size, num_classes, shared_layers, task_layers, self.num_tasks).to(device)
        

        self.optimizer_mu = optim.Adam(self.mu_net.parameters(), lr=self.learning_rate)

        self.criterion_classification = nn.CrossEntropyLoss()

        self.model_train_losses = np.zeros((self.epochs, self.num_tasks))
        self.model_train_accuracies = np.zeros((self.epochs, self.num_tasks)) 

        logging.info("Initialized DCNPDLearner on device: %s", device)

    def _get_loader(self, X, y, w):
        dataset = CustomDataset(X, y, w)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def _compute_ipw_weights(self, X, treated=True):
        prop_scores = self.predict_propensity(X)
        if not treated:
            prop_scores = 1 - prop_scores
        return 1.0 / np.clip(prop_scores, 1e-5, 1)

    def train_multi_task(self, X_control, y_control, w_control, X_treated, y_treated, w_treated, task_type="classification"):
        loader_control = self._get_loader(X_control, y_control, w_control)
        loader_treated = self._get_loader(X_treated, y_treated, w_treated)

        model = self.mu_net if task_type == "classification" else self.tau_net
        optimizer = self.optimizer_mu if task_type == "classification" else self.optimizer_tau
        criterion = self.criterion_classification if task_type == "classification" else self.criterion_regression

        model.train()
        total_loss, total_c, total_t = 0.0, 0.0, 0.0
        correct_c, correct_t = 0, 0
        n_c, n_t = 0, 0

        ps_score_control =  torch.tensor(self.predict_propensity(X_control), dtype=torch.float32).to(self.device)
        ps_score_treated =  torch.tensor(self.predict_propensity(X_treated), dtype=torch.float32).to(self.device)

        entropy_control = shannon_entropy(ps_score_control).to(self.device)
        entropy_treated = shannon_entropy(ps_score_treated).to(self.device)

        dropout_prob_control = dropout_probability(entropy_control, gamma=1).to(self.device)
        dropout_prob_treated = dropout_probability(entropy_treated, gamma=1).to(self.device)

        for i, ((Xc, yc, _), (Xt, yt, _)) in enumerate(zip(loader_control, loader_treated)):
            Xc, Xt = Xc.float().to(self.device), Xt.float().to(self.device)
            yc = (yc.long() if task_type == "classification" else yc.float().view(-1, 1)).to(self.device)
            yt = (yt.long() if task_type == "classification" else yt.float().view(-1, 1)).to(self.device)

            m0_out, m1_out = model(Xc,dropout_prob_control[i * self.batch_size:(i + 1) * self.batch_size])
            _, m1_out_ = model(Xt,dropout_prob_treated[i * self.batch_size:(i + 1) * self.batch_size])

            loss_c = criterion(m0_out, yc).mean()
            loss_t = criterion(m1_out_, yt).mean()

            loss = loss_c + loss_t 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_c += loss_c.item()
            total_t += loss_t.item()
            total_loss += loss.item()

            if task_type == "classification":
                correct_c += (torch.argmax(m0_out, 1) == yc).sum().item()
                correct_t += (torch.argmax(m1_out_, 1) == yt).sum().item()
                n_c += yc.size(0)
                n_t += yt.size(0)
            else:
                correct_c += ((m0_out - yc) ** 2).sum().item()
                correct_t += ((m1_out_ - yt) ** 2).sum().item()
                n_c += yc.size(0)
                n_t += yt.size(0)

        return total_c / len(loader_control), total_t / len(loader_treated), correct_c / n_c, correct_t / n_t

    def fit(self, X, y, t_ind):
        X, y, t_ind = np.array(X), np.array(y), np.array(t_ind)

        logging.info("Training response functions (mu0, mu1)...")
        for epoch in range(self.epochs):
            loss_c, loss_t, acc_c, acc_t = self.train_multi_task(
                X[t_ind == 0], y[t_ind == 0], t_ind[t_ind == 0],
                X[t_ind == 1], y[t_ind == 1], t_ind[t_ind == 1],
            )
            self.model_train_losses[epoch] = [loss_c, loss_t]
            self.model_train_accuracies[epoch] = [acc_c, acc_t]
            logging.info(f"[Epoch {epoch+1}] Loss: {loss_c+loss_t:.4f}, Acc (C): {acc_c:.4f}, Acc (T): {acc_t:.4f}")


        return self

    def predict_propensity(self, X, batch_size=None, class_index=1):
        """
        Predicts propensity scores P(W=1|X) for a given feature matrix X.
        Assumes the model outputs logits for binary classification.

        Args:
            X (np.ndarray): Feature matrix of shape (N, d)
            batch_size (int, optional): Optional batching for memory efficiency
        
        Returns:
            np.ndarray: Propensity scores of shape (N,)
        """
        self.propensity_model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        y_pred_proba = []

        with torch.no_grad():
            if batch_size:
                for i in range(0, len(X), batch_size):
                    batch = X_tensor[i:i + batch_size]
                    outputs = self.propensity_model(batch)
                    probs = torch.softmax(outputs, dim=1)[:, class_index]  # P(W=1|X)
                    y_pred_proba.extend(probs.cpu().numpy())
            else:
                outputs = self.propensity_model(X_tensor)
                probs = torch.softmax(outputs, dim=1)[:, class_index]
                y_pred_proba = probs.cpu().numpy()

        return np.clip(np.array(y_pred_proba), 1e-5, 1 - 1e-5)  # For numerical stability

    def predict_outcome(self, X, y, t_ind, treated_flag=True):
        """
        Predict outcomes for either the treated or control group using the appropriate model.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Ground truth labels (not used during prediction).
            t_ind (np.ndarray): Treatment indicator.
            treated_flag (bool): Whether to predict for treated (1) or control (0) group.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: y_true, y_pred, y_pred_proba
        """
        self.mu_net.eval()

        # Prepare DataLoader
        dataset = CustomDataset(X, y, t_ind)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        y_true, y_pred, y_pred_proba = [], [], []

        with torch.no_grad():
            for X_batch, y_batch, t_batch in loader:
                X_batch = X_batch.float().to(self.device)
                y_batch = y_batch.long().to(self.device)

                m0_out, m1_out = self.mu_net(X_batch)
                logits = m1_out if treated_flag else m0_out

                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                y_pred_proba.extend(probs.cpu().numpy())
                y_pred.extend(preds.cpu().tolist())
                y_true.extend(y_batch.cpu().tolist())
                

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
        torch.save(self.mu_net.state_dict(), os.path.join(path, "mu_net.pth"))

        # Save config/state (excluding models)
        joblib.dump({
            'input_size': self.input_size,
            'num_classes': self.num_classes,
            'params': self.params,
            'device': self.device,
            'epochs': self.epochs,
            'model_train_losses': self.model_train_losses,
            'model_train_accuracies': self.model_train_accuracies,
        }, os.path.join(path, "dcnpdlearner_config.pkl"))
        
        logging.info(f"✅ DCNPDLearner saved to {path}")
    
    def load(self, path, device):
        # Load config
        state = joblib.load(os.path.join(path, "dcnpdlearner_config.pkl"))

        # Override the training device based on the testing device
        self.device = device

        # Restore attributes
        self.input_size = state['input_size']
        self.num_classes = state['num_classes']
        self.params = state['params']
        self.epochs = state['epochs']
        self.model_train_losses = state['model_train_losses']
        self.model_train_accuracies = state['model_train_accuracies']

        # Rebuild model architecture
        self.mu_net = DCNet(
            self.input_size, self.params['hidden_size'], self.num_classes,
            self.params['num_shared_layers'], self.params['num_task_specific_layers'], self.params['num_tasks']
        ).to(self.device)

        # Load weights
        self.mu_net.load_state_dict(torch.load(os.path.join(path, "mu_net.pth"), map_location=self.device))
        logging.info("Learner loaded")
