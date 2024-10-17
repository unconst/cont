import torch
from torch.utils.data import Dataset


class DatasetFactory:
    @staticmethod
    def create_dataset(model_type, num_samples, num_features):
        match model_type:
            case "tiny_nn":
                return LinearRegressionDataset(num_samples, num_features)
            case _:
                raise ValueError(f"Unsupported model type: {model_type}")


class LinearRegressionDataset(Dataset):
    def __init__(self, num_samples, num_features):
        # Generate features and targets
        self.features, self.targets = self._generate_linear_data(
            num_samples, num_features
        )

    def _generate_linear_data(self, num_samples, num_features):
        # Generate random features
        X = torch.rand(num_samples, num_features)
        # Generate random weights for linear relationship
        weights = torch.rand(num_features)
        # Generate targets with some noise
        y = X @ weights + torch.randn(num_samples) * 0.1
        return X, y

    def __len__(self):
        # Return the total number of samples
        return len(self.features)

    def __getitem__(self, idx):
        # Return a single sample (features and corresponding target)
        return self.features[idx], self.targets[idx]
