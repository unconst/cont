from abc import ABC, abstractmethod


class LossCalculator(ABC):
    @abstractmethod
    def calculate(self, validator_model, miner_slice, data):
        pass


class ExactLoss(LossCalculator):
    def calculate(self, validator_model, miner_slice, data):
        # Exact forward pass loss calculation
        loss_before = validator_model.compute_loss(data)
        # Apply miner slice
        validator_model.update_params(miner_slice)
        loss_after = validator_model.compute_loss(data)
        return loss_after - loss_before


class FirstOrderTaylor(LossCalculator):
    def calculate(self, validator_model, miner_slice, data):
        raise NotImplementedError()


class SecondOrderTaylor(LossCalculator):
    def calculate(self, validator_model, miner_slice, data):
        raise NotImplementedError()


class CosineSimilarity(LossCalculator):
    def calculate(self, validator_model, miner_slice, data):
        raise NotImplementedError()
