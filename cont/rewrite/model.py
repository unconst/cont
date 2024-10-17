import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


from cont.hparams import load_hparams

from loguru import logger


class ModelFactory:
    @staticmethod
    def create_model(model_type: str):
        if model_type == "tiny_nn":
            return Model(TinyNeuralNetwork(2, 3, 1))
        else:
            raise ValueError(f"Unsupported model type: {model_type}")


class TinyNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=10, output_size=1):
        super(TinyNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class Model:
    def __init__(self, model):
        self.hparams = load_hparams()
        self.model = model
        self.criterion = nn.MSELoss()  # Mean squared error for regression
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.learning_rate,  # Peak learning rate
            betas=(
                self.hparams.optimizer_beta1,
                self.hparams.optimizer_beta2,
            ),  # B1 and B2
            weight_decay=self.hparams.optimizer_weight_decay,  # Weight decay
            foreach=True,  # more memory usage, but faster
        )

    def train_step(self, data):
        inputs, targets = data
        inputs, targets = (
            torch.tensor(inputs, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.float32),
        )

        # Zero gradients from previous step
        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.model(inputs)

        # Compute loss
        loss = self.criterion(outputs, targets)

        # Backward pass (compute gradients)
        loss.backward()

        # Update weights
        self.optimizer.step()

        return loss.item()

    def get_params(self):
        # Get a dictionary of model parameters
        return {name: param.clone() for name, param in self.model.named_parameters()}

    def update_params(self, param_slice):
        # Update model parameters using received slice
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in param_slice:
                    param.copy_(param_slice[name])

    def compute_loss(self, data):
        # Compute loss for validation
        inputs, targets = data
        inputs, targets = (
            torch.tensor(inputs, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.float32),
        )

        with torch.no_grad():
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

        return loss.item()


def sum_params(m: TinyNeuralNetwork):
    return [p.sum() for p in m.parameters()]


def num_params(m: TinyNeuralNetwork):
    return sum(p.numel() for p in m.parameters())


def sample_parameters():
    # Set all parameters to zero
    torch_model = ModelFactory.create_model("tiny_nn").model
    for param in torch_model.parameters():
        param.data.zero_()
    logger.info(f"Sum of each group of params: {sum_params(torch_model)=}")

    # GET SLICE
    # Flatten parameters
    params = torch.nn.utils.parameters_to_vector(torch_model.parameters())
    logger.info(f"Params shape: {params.shape}")
    logger.info(
        f"Number of elements in each group of params: {[p.numel() for p in torch_model.parameters()]}"
    )

    # Select random param indices
    num_params_total = num_params(torch_model)
    compression_factor = 4
    slice_size = num_params(torch_model) // compression_factor
    random_indices = torch.tensor(
        np.random.choice(num_params_total, slice_size, replace=False)
    )
    logger.info(f"Total number of parameters: {num_params_total}")
    logger.info(f"Number of selected indices: {random_indices}")

    # Create slice
    slice = torch.ones_like(random_indices)

    # Average only at random_indices
    params[random_indices] = (params[random_indices] + slice) / 2

    # Assign the averaged parameters back to the model
    torch.nn.utils.vector_to_parameters(params, torch_model.parameters())
    logger.info(f"{sum_params(torch_model)=}")
    print("\n")


for _ in range(10):
    sample_parameters()
