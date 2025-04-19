import torch
import torch.nn as nn
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, X, y, w):
        self.X = X if isinstance(X, torch.Tensor) else torch.tensor(X)
        self.y = y if isinstance(y, torch.Tensor) else torch.tensor(y)
        self.w = w if isinstance(w, torch.Tensor) else torch.tensor(w)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.w[idx]

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size, num_layers, dropout_rate):
        super(NeuralNetwork, self).__init__()

        layers = []
        prev_size = input_size  # Initial input size

        for i in range(num_layers):  # Add layers dynamically
            next_size = max(hidden_size // (2 ** i), 16)  # Avoid zero-sized layers
            layers.append(nn.Linear(prev_size, next_size))
            layers.append(nn.BatchNorm1d(next_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = next_size  # Update for next layer

        layers.append(nn.Linear(prev_size, num_classes))  # Output layer
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class MultiTaskNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate, num_shared_layers, num_task_specific_layers, num_tasks=2):
        super(MultiTaskNet, self).__init__()
        self.num_tasks = num_tasks
        self.shared_layers = nn.ModuleList()
        current_size = input_size
        
        # === Shared layers ===
        for _ in range(num_shared_layers):
            self.shared_layers.append(nn.Linear(current_size, hidden_size))
            self.shared_layers.append(nn.BatchNorm1d(hidden_size))
            self.shared_layers.append(nn.ReLU())
            self.shared_layers.append(nn.Dropout(dropout_rate))
            current_size = hidden_size

        # === Task-specific heads ===
        self.task_layers = nn.ModuleList()  # each element: nn.Sequential for one task

        for _ in range(num_tasks):
            task_layer_list = []
            task_input_size = current_size

            for _ in range(num_task_specific_layers):
                task_layer_list.append(nn.Linear(task_input_size, task_input_size // 2))
                task_layer_list.append(nn.BatchNorm1d(task_input_size // 2))
                task_layer_list.append(nn.ReLU())
                task_layer_list.append(nn.Dropout(dropout_rate))
                task_input_size = task_input_size // 2

            # Final output layer for this task
            task_layer_list.append(nn.Linear(task_input_size, num_classes))
            self.task_layers.append(nn.Sequential(*task_layer_list))

    def forward(self, x):
        # Shared encoding
        for layer in self.shared_layers:
            x = layer(x)

        # Task-specific outputs
        outputs = [head(x) for head in self.task_layers]  # list of size `num_tasks`
        return outputs

class DCNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_shared_layers, num_task_specific_layers, num_tasks=2):
        super(DCNet, self).__init__()
        self.num_tasks = num_tasks
        self.shared_layers = nn.ModuleList()
        current_size = input_size
        
        # === Shared layers ===
        for _ in range(num_shared_layers):
            self.shared_layers.append(nn.Linear(current_size, hidden_size))
            self.shared_layers.append(nn.BatchNorm1d(hidden_size))
            self.shared_layers.append(nn.ReLU())
            current_size = hidden_size

        # === Task-specific heads ===
        self.task_layers = nn.ModuleList()  # each element: nn.Sequential for one task

        for _ in range(num_tasks):
            task_layer_list = []
            task_input_size = current_size

            for _ in range(num_task_specific_layers):
                task_layer_list.append(nn.Linear(task_input_size, task_input_size // 2))
                task_layer_list.append(nn.BatchNorm1d(task_input_size // 2))
                task_layer_list.append(nn.ReLU())
                task_input_size = task_input_size // 2

            # Final output layer for this task
            task_layer_list.append(nn.Linear(task_input_size, num_classes))
            self.task_layers.append(nn.Sequential(*task_layer_list))

    def forward(self, x, drop_prob=None):
        """
        drop_prob: (batch,) tensor of per-sample dropout probabilities.
                   If None, no dropout is applied.
        """
        h = x
        for i, layer in enumerate(self.shared_layers):
            if isinstance(layer, nn.Linear):
                h = layer(h)
                if drop_prob is not None:
                    mask = self.dropout_mask(h, drop_prob)
                    h = mask * h
            elif isinstance(layer, nn.BatchNorm1d):
                h = layer(h)
            elif isinstance(layer, nn.ReLU):
                h = layer(h)


        outputs = []
        for task_head in self.task_layers:
            h_task = h  # shared representation (after dropout if training)
            for layer in task_head:
                if isinstance(layer, nn.Linear):
                    h_task = layer(h_task)
                    if drop_prob is not None:
                        mask = self.dropout_mask(h_task, drop_prob)
                        h_task = mask * h_task
                elif isinstance(layer, nn.BatchNorm1d):
                    h_task = layer(h_task)
                elif isinstance(layer, nn.ReLU):
                    h_task = layer(h_task)

            outputs.append(h_task)

        return outputs
    
    def dropout_mask(self, x, prob):
        """
        x: (batch, features)
        prob: (batch,) tensor with values in [0, 1)
        """
        # print("Dropout prob", prob)
        eps = 1e-6
        keep_prob = (1.0 - prob).clamp(min=eps, max=1.0 - eps)  # shape: (batch,)
        keep_prob = keep_prob.unsqueeze(1).expand_as(x)         # shape: (batch, features)
        
        bernoulli = torch.distributions.Bernoulli(probs=keep_prob)
        mask = bernoulli.sample().to(x.device)
        return mask / keep_prob