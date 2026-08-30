# %% Assignment
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# %% 1. Target function and its exact analytical derivative
def target_function(x):
    return torch.sin(2 * torch.pi * x) + 0.5 * x

def target_derivative(x):
    return 2 * torch.pi * torch.cos(2 * torch.pi * x) + 0.5

# %% 2. Neural Network Architecture
class SurrogateNet(nn.Module):
    def __init__(self):
        super(SurrogateNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

# %% 3. Training Setup
model = SurrogateNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

x_train = torch.linspace(-1, 1, 200).unsqueeze(1)
y_train = target_function(x_train)

# %% 4. Training Loop (Tracking both function loss and derivative loss per epoch)
epochs = 500
function_loss_history = []
derivative_loss_history = []

for epoch in range(epochs):
    # Enable gradient tracking on input to compute derivative loss during training
    x_train_grad = x_train.clone().detach().requires_grad_(True)
    
    optimizer.zero_grad()
    y_pred = model(x_train_grad)
    
    # Compute Function Loss
    loss_func = criterion(y_pred, target_function(x_train_grad))
    
    # Compute Autograd Derivative & Derivative Loss
    dy_dx_pred = torch.autograd.grad(
        outputs=y_pred, 
        inputs=x_train_grad, 
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True
    )[0]
    
    loss_deriv = criterion(dy_dx_pred, target_derivative(x_train_grad))
    
    # Backpropagation on function loss
    loss_func.backward()
    optimizer.step()
    
    function_loss_history.append(loss_func.item())
    derivative_loss_history.append(loss_deriv.item())

# %% 5. Test Evaluation Grid
x_test = torch.linspace(-2, 2, 600).unsqueeze(1).requires_grad_(True)
y_test_pred = model(x_test)
y_test_true = target_function(x_test)

dy_dx_test_pred = torch.autograd.grad(
    outputs=y_test_pred, 
    inputs=x_test, 
    grad_outputs=torch.ones_like(y_test_pred)
)[0]
dy_dx_test_true = target_derivative(x_test)

# Detach tensors for matplotlib
x_np = x_test.detach().numpy()
y_true_np = y_test_true.detach().numpy()
y_pred_np = y_test_pred.detach().numpy()
dy_true_np = dy_dx_test_true.detach().numpy()
dy_pred_np = dy_dx_test_pred.detach().numpy()

# %% 6. Generate 2x2 Subplot Figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot (0,0): Function vs. Surrogate
axes[0, 0].plot(x_np, y_true_np, label="True f(x)", color="black", linewidth=2)
axes[0, 0].plot(x_np, y_pred_np, label="Surrogate ŷ(x)", color="red", linestyle="--")
axes[0, 0].set_title("1. Function Approximation: y vs. x")
axes[0, 0].set_xlabel("x")
axes[0, 0].set_ylabel("y")
axes[0, 0].legend()
axes[0, 0].grid(True)

# Plot (0,1): Derivatives Comparison
axes[0, 1].plot(x_np, dy_true_np, label="True f'(x)", color="black", linewidth=2)
axes[0, 1].plot(x_np, dy_pred_np, label="Autograd dŷ/dx", color="blue", linestyle="--")
axes[0, 1].set_title("2. Spatial Derivative: dy/dx vs. x")
axes[0, 1].set_xlabel("x")
axes[0, 1].set_ylabel("dy/dx")
axes[0, 1].legend()
axes[0, 1].grid(True)

# Plot (1,0): Function Loss History
axes[1, 0].plot(function_loss_history, color="purple")
axes[1, 0].set_title("3. Function Loss vs. Epochs")
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("MSE Loss")
axes[1, 0].grid(True)

# Plot (1,1): Derivative Loss History
axes[1, 1].plot(derivative_loss_history, color="green")
axes[1, 1].set_title("4. Derivative Loss vs. Epochs")
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("MSE Loss")
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()
# %%
