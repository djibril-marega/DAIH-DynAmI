"""Deep Active Inference Hybrid framework."""
__version__ = "4.0.0"

from .agent import Agent_DAIH
from .offline_training.offline import OfflineConfig

__all__ = ["Agent_DAIH", "OfflineConfig", "__version__"]
