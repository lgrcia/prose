from prose.modeling.base_models import Model
import numpy as np

class ConstantModel(Model):
    """
    A simple concrete model with a single parameter ``value``
    Args:
        value (float): The value of the model.
    """

    parameter_names = ("value", )

    def get_value(self, x):
        return self.value + np.zeros(len(x))

    def compute_gradient(self, x):
        return np.ones((1, len(x)))
        
class Protopapas2005(Model):
    parameter_names = ("t0", "duration", "depth", "c", "period")
    
    def get_value(self, t): 
        _t = self.period * np.sin(np.pi * (t - self.t0) / self.period) / (np.pi * self.duration)
        return (1 - self.depth) + (self.depth / 2) * (
            2 - np.tanh(self.c * (_t + 1 / 2)) + np.tanh(self.c * (_t - 1 / 2))
        )