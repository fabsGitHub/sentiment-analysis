import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import scipy.sparse as sp

class PyTorchMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_classes=3, hidden_layer_sizes=(64,), activation='relu', solver='adam',
                 alpha=0.0001, batch_size=32, learning_rate_init=0.001,
                 max_iter=100, early_stopping=True, validation_fraction=0.1,
                 patience=10, random_state=42):
        self.n_classes = n_classes
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.random_state = random_state

    def fit(self, X, y):
        # 1. Device selection
        if torch.cuda.is_available():
            self.device_ = torch.device("cuda")
            use_speedups = True
        elif torch.backends.mps.is_available():
            self.device_ = torch.device("mps")
            use_speedups = False # Deactivate pinning/non-blocking on Apple Silicon
        else:
            self.device_ = torch.device("cpu")
            use_speedups = False

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        # 2. Dynamic class matching
        unique_classes = np.unique(y)
        self.runtime_classes_ = len(unique_classes)

        if hasattr(X, "toarray"):
            X = X.toarray()
        if sp.issparse(X):
            X = X.toarray()

        X_tensor = torch.tensor(X, dtype=torch.float32)
        # Ensure target labels are dense, continuous, and aligned
        y_tensor = torch.tensor(np.array(y), dtype=torch.long)

        # 3. Build Network Architecture
        input_dim = X.shape[1]
        layers = []
        prev_dim = input_dim
        for h_size in self.hidden_layer_sizes:
            layers.append(nn.Linear(prev_dim, h_size))
            if self.activation == 'relu':
                layers.append(nn.ReLU())
            elif self.activation == 'tanh':
                layers.append(nn.Tanh())
            prev_dim = h_size

        layers.append(nn.Linear(prev_dim, self.runtime_classes_))
        self.model_ = nn.Sequential(*layers).to(self.device_)

        # 4. Train / Validation Split
        if self.early_stopping and self.validation_fraction > 0:
            val_size = int(len(X) * self.validation_fraction)
            if val_size > 0:
                X_train, X_val = X_tensor[:-val_size], X_tensor[-val_size:]
                y_train, y_val = y_tensor[:-val_size], y_tensor[-val_size:]
                val_dataset = TensorDataset(X_val, y_val)
                val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=use_speedups)
            else:
                X_train, y_train = X_tensor, y_tensor
                val_loader = None
        else:
            X_train, y_train = X_tensor, y_tensor
            val_loader = None

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=use_speedups)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate_init, weight_decay=self.alpha)

        best_loss = float('inf')
        epochs_no_improve = 0

        # Ensure max_iter is treated cleanly as an integer scalar
        total_epochs = int(self.max_iter)

        # 5. Training Loop
        for epoch in range(total_epochs):
            self.model_.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device_, non_blocking=use_speedups)
                batch_y = batch_y.to(self.device_, non_blocking=use_speedups)

                optimizer.zero_grad()
                outputs = self.model_(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=1.0)
                optimizer.step()

            # 6. Evaluation Hook
            if val_loader is not None:
                self.model_.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device_, non_blocking=use_speedups)
                        batch_y = batch_y.to(self.device_, non_blocking=use_speedups)
                        outputs = self.model_(batch_x)
                        val_loss += criterion(outputs, batch_y).item()
                val_loss /= len(val_loader)

                if val_loss < best_loss:
                    best_loss = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        break
        return self

    def predict(self, X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        if sp.issparse(X):
            X = X.toarray()

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device_)
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(X_tensor)
            _, preds = torch.max(outputs, 1)
        return preds.cpu().numpy()
