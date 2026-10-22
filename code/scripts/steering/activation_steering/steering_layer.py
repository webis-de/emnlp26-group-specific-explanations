import torch
import logging

from typing import Optional

logger = logging.getLogger(__name__)


class SteeringLayer(torch.nn.Module):
    """
    A layer that applies a steering vector to the output of a given layer.
    The steering vector is added to the output of the layer, scaled by a lambda value.
    The steering vector is expected to be of shape (1, `config.STEERING_MODEL_ACTIVATION_VECTOR_SIZE`).
    """

    def __init__(
        self,
        layer: torch.nn.Module,
        steering_vector: Optional[torch.Tensor] = None,
        lmbda: float = 1.0,
    ):
        """
        Initialize the SteeringLayer.
        Args:
            layer (torch.nn.Module): The layer to be steered.
            steering_vector (Optional[torch.Tensor]): The steering vector to be applied. If None, it must be set later. It has to be of shape (1, `config.STEERING_MODEL_ACTIVATION_VECTOR_SIZE`).
            lmbda (float): The scaling factor for the steering vector.
        """
        super().__init__()
        self.layer = layer
        self.device = layer.parameters().__next__().device
        self.lmbda = lmbda
        if steering_vector is not None:
            self.set_steering_vector(steering_vector)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._steering_vector_set:
            logger.error("Steering vector is not set, cannot apply steering.")
            raise ValueError("Steering vector is not set.")
            return self.layer(x)
        return self.layer(x) + self.lmbda * self._steering_vector

    def set_steering_vector(self, steering_vector: torch.Tensor):
        """
        Set the steering vector for the layer.

        Args:
            steering_vector (torch.Tensor): The steering vector to set
        """

        self._steering_vector = steering_vector.to(self.device)
        self._steering_vector_set = True
