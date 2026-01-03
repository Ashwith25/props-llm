import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

class NNBrainReward:
    def __init__(self, input_dim, lr=1e-3, epochs=50, hidden_dim=64, device="cpu"):
        self.input_dim = input_dim
        self.epochs = epochs
        self.device = device
        self.model = MLP(input_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        # Statistics for normalization
        self.x_mean = 0
        self.x_std = 1
        self.y_mean = 0
        self.y_std = 1

    def train(self, replay_buffer):
        """
        Retrains the network from scratch (or fine-tunes) on the entire replay buffer.
        """
        if len(replay_buffer.buffer) < 10:
            return # Not enough data to train effectively

        # 1. Extract Data
        # Buffer stores: (params, true_reward, pred_reward, confidence)
        X = np.array([item[0].flatten() for item in replay_buffer.buffer])
        y = np.array([item[1] for item in replay_buffer.buffer]).reshape(-1, 1)

        # 2. Update Normalization Stats
        self.x_mean = np.mean(X, axis=0)
        self.x_std = np.std(X, axis=0) + 1e-8 # Avoid div by zero
        self.y_mean = np.mean(y)
        self.y_std = np.std(y) + 1e-8

        # 3. Normalize
        X_norm = (X - self.x_mean) / self.x_std
        y_norm = (y - self.y_mean) / self.y_std

        # 4. Convert to Tensors
        X_tensor = torch.FloatTensor(X_norm).to(self.device)
        y_tensor = torch.FloatTensor(y_norm).to(self.device)

        # 5. Training Loop
        self.model.train()
        for _ in range(self.epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()

    def predict(self, params):
        """
        Predicts reward for a single parameter vector.
        """
        self.model.eval()
        
        # Prepare Input
        params_flat = np.array(params).flatten()
        params_norm = (params_flat - self.x_mean) / self.x_std
        params_tensor = torch.FloatTensor(params_norm).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred_norm = self.model(params_tensor).item()
        
        # Denormalize Output
        pred_reward = (pred_norm * self.y_std) + self.y_mean
        
        # Baseline Confidence: NN doesn't give confidence natively unless using Ensemble/Dropout.
        # We return a dummy high confidence to indicate it's a deterministic guess.
        confidence = 1.0 
        
        # Generate a dummy reasoning string for logging compatibility
        reasoning = f"Neural Network Prediction based on {self.input_dim} parameters."

        return pred_reward, confidence, reasoning