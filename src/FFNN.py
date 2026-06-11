
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class PyTorchMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_classes=3, hidden_layer_sizes=(64,), activation='relu', solver='adam',
                 alpha=0.0001, batch_size=32, learning_rate_init=0.001,
                 max_iter=5, early_stopping=True, validation_fraction=0.1, random_state=42):
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
        self.random_state = random_state

    def fit(self, X, y):
        if torch.backends.mps.is_available():
            self.device_ = torch.device("mps")
        elif torch.cuda.is_available():
            self.device_ = torch.device("cuda")
        else:
            self.device_ = torch.device("cpu")

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        if hasattr(X, "toarray"): X = X.toarray()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(np.array(y), dtype=torch.long)

        self.classes_ = np.unique(y)
        num_classes = len(self.classes_)
        input_dim = X_tensor.shape[1]

        layers = []
        prev_dim = input_dim
        for hidden_dim in self.hidden_layer_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if self.activation == 'relu': layers.append(nn.ReLU())
            elif self.activation == 'tanh': layers.append(nn.Tanh())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))

        self.model_ = nn.Sequential(*layers).to(self.device_)
        criterion = nn.CrossEntropyLoss()

        if self.solver == 'adam':
            optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate_init, weight_decay=self.alpha)
        else:
            optimizer = optim.SGD(self.model_.parameters(), lr=self.learning_rate_init, weight_decay=self.alpha)

        dataset = TensorDataset(X_tensor, y_tensor)

        if self.early_stopping and self.validation_fraction > 0:
            val_size = int(len(dataset) * self.validation_fraction)
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            train_dataset = dataset
            val_loader = None

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        best_loss = float('inf')
        epochs_no_improve = 0
        patience = 3

        for epoch in range(self.max_iter):
            self.model_.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device_), batch_y.to(self.device_)
                optimizer.zero_grad()
                outputs = self.model_(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            if val_loader is not None:
                self.model_.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x, batch_y = batch_x.to(self.device_), batch_y.to(self.device_)
                        outputs = self.model_(batch_x)
                        val_loss += criterion(outputs, batch_y).item()
                val_loss /= len(val_loader)

                if val_loss < best_loss:
                    best_loss = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience: break
        return self

    def predict(self, X):
        if hasattr(X, "toarray"): X = X.toarray()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device_)
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(X_tensor)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()
