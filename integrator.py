# Adopted from NeuroMANCER (https://pnnl.github.io/neuromancer/_modules/integrators.html)
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class Integrator(nn.Module, ABC):

    def __init__(self, block, h=1.0):
        """
        Integration block.

        :param block:
        :param h:
        """
        super().__init__()
        self.block = block  # block gives dx at this time step.
        self.in_features, self.out_features = block.in_features, block.out_features
        self.h = h
        self.state = lambda x, tq, t, u: torch.hstack([x, u])

    @abstractmethod
    def integrate(self, x, u, t, matA, matB):  # YK added x, u, t, matA, matB
        pass

    def forward(self, x, u=torch.empty((1, 1)), t=torch.empty((1, 1)), matA=None, matB=None):
        """
        This function needs x only for autonomous systems. x is 2D.
        It needs both x and t for nonautonomous system w/ offline interpolation. Both x and t are 2D.
        It needs all x, t and u for nonautonomous system w/ online interpolation. x is 2D while both t and u are 3D.
        """
        return self.integrate(x, u, t, matA, matB)

    def reg_error(self):
        return sum([k.reg_error() for k in self.children() if hasattr(k, "reg_error")])


class RK4(Integrator):
    def __init__(self, block, h=1.0):
        """
        :param block: (nn.Module) A state transition model.
        :param h: (float) integration step size
        """
        super().__init__(block=block, h=h)

    def integrate(self, x, u, t, matA, matB):
        h = self.h
        k1 = self.block(self.state(x, t, t, u), matA, matB)  # k1 = f(x_i, t_i)
        k2 = self.block(self.state(x+0.5*h*k1, t+0.5*h, t, u), matA, matB)  # k2 = f(x_i + 0.5*h*k1, t_i + 0.5*h)
        k3 = self.block(self.state(x+0.5*h*k2, t+0.5*h, t, u), matA, matB)  # k3 = f(x_i + 0.5*h*k2, t_i + 0.5*h)
        k4 = self.block(self.state(x+h*k3, t+h, t, u), matA, matB)  # k4 = f(x_i + h*k3, t_i + h)
        return x + h * (k1/6.0 + k2/3.0 + k3/3.0 + k4/6.0)

class Naive(Integrator):
    def __init__(self, block, h=1.0):
        """
        :param block: (nn.Module) A state transition model.
        :param h: (float) integration step size
        """
        super().__init__(block=block, h=h)

    def integrate(self, x, u, t, matA, matB):
        dx = self.block(self.state(x, t, t, u), matA, matB)
        return x + self.h * dx
