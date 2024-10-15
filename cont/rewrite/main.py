from rewrite.miner import Miner
from rewrite.validator import Validator
from rewrite.model import ModelFactory
from rewrite.dataset import DatasetFactory
from rewrite.loss import (
    ExactLoss,
)
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from copy import deepcopy
from loguru import logger
import itertools

# Parameters
SAME_MODEL = True
num_miners = 3
num_communication_rounds = 10
input_size = 10  # Number of features for regression
hidden_size = 10
output_size = 1
compression_factor = 4
training_data_fraction = 0.1
data_size = 1000
slice_size = input_size // compression_factor
model_type = "tiny_nn"  # Can be "neural_network" or future LLM


# Initialize miners
def get_training_indices() -> torch.Tensor:
    return torch.randint(0, data_size, (int(training_data_fraction * data_size),))


# Generate the appropriate dataset for the model type
dataset = DatasetFactory.create_dataset(model_type, data_size, input_size)
logger.success(f"Created dataset for model {model_type}")

with torch.random.fork_rng():
    if SAME_MODEL:
        torch.manual_seed(42)
    miner_models = num_miners * [ModelFactory.create_model(model_type)]
logger.success(f"Created {len(miner_models)} miner models")


miners = [
    Miner(
        model,
        (
            dataset.features[get_training_indices()],
            dataset.targets[get_training_indices()],
        ),
        i,
    )
    for i, model in enumerate(miner_models)
]
logger.success(f"Created {len(miners)} miners")
logger.debug(miners[0].data)

# Choose the loss method dynamically using dependency injection
loss_method = (
    ExactLoss()
)  # Change this to FirstOrderTaylor, SecondOrderTaylor, or CosineSimilarity
validator = Validator(
    deepcopy(miner_models[0]),
    slice_size,
    compression_factor,
    loss_method,
)
logger.success("Created validator")

# Create the DataLoader with shuffle
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Create an infinite iterator
infinite_loader = itertools.cycle(loader)
# Training loop
for round_num in range(num_communication_rounds):
    print(f"Communication Round {round_num + 1}")

    model_slices = []
    for miner in miners:
        miner.data = next(infinite_loader)
        logger.info("Training one step")
        miner.train_one_step()
        slice_indices = validator.request_slice_indices(input_size)
        model_slice = miner.send_model_slice(slice_indices)
        model_slices.append(model_slice)

    validator.aggregate_slices(model_slices)

    for miner in miners:
        miner_score = validator.calculate_miner_score(
            miner.model, model_slices[miner.miner_id], miner.data, method=loss_method
        )
        print(f"Miner {miner.miner_id} score: {miner_score}")

        new_model_slice = validator.send_aggregated_slice(slice_size)
        miner.receive_model_slice(new_model_slice)
