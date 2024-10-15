import random
import numpy as np


class Validator:
    def __init__(self, model, num_slices, compression_factor, loss_calculator):
        self.model = model
        self.num_slices = num_slices
        self.compression_factor = compression_factor
        self.loss_calculator = loss_calculator

    def request_slice_indices(self, model_size):
        # Randomly select parameter indices based on compression factor
        return random.sample(range(model_size), model_size // self.compression_factor)

    def aggregate_slices(self, model_slices):
        # Aggregate slices from miners
        aggregated_params = {}
        for param_idx in model_slices[0].keys():
            aggregated_params[param_idx] = np.mean(
                [miner_slice[param_idx] for miner_slice in model_slices], axis=0
            )
        self.model.update_params(aggregated_params)

    def calculate_miner_score(self, miner_model, miner_slice, data, method):
        # Use the loss calculator to compute the score
        return self.loss_calculator.calculate(miner_model, miner_slice, data, method)

    def send_aggregated_slice(self, slice_size):
        # Send slice of the averaged model to miners
        model_params = self.model.get_params()
        return {
            idx: model_params[idx]
            for idx in random.sample(range(len(model_params)), slice_size)
        }
