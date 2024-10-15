class Miner:
    def __init__(self, model, data, miner_id):
        self.model = model
        self.data = data
        self.miner_id = miner_id

    def train_one_step(self):
        # Perform one training step (forward, backward, update)
        self.model.train_step(self.data)

    def send_model_slice(self, params_indices):
        # Get model parameters
        for name, param in self.model.model.named_parameters():
            print(f"{name=}, {param.shape=}")
        exit()
        model_params = self.model.get_params()
        print(f"{model_params.shape=}")
        print(f"{params_indices=}, {len(params_indices)=}")

        # Map numeric indices to parameter names
        param_names = list(model_params.keys())
        print(f"{param_names=}")

        # Ensure params_indices contains valid indices
        slice_params = {
            param_names[idx]: model_params[param_names[idx]] for idx in params_indices
        }

        return slice_params

    def receive_model_slice(self, slice):
        # Update part of the model with received slice from validator
        self.model.update_params(slice)
