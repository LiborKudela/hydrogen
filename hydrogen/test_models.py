"""Synthetic ODE models used to validate the time integrator end-to-end."""

from __future__ import annotations

import numpy as np

from .model import DifferentialVariable, Model, Parameter, Variable


class InnerODE_1(Model):
    """`der_variable = p * dummy`, with `variable` linked to `der_variable` via Crank-Nicolson."""

    def __init__(self):
        super().__init__()

    def declare_components(self):
        self.add_component('p', Parameter(1, "m/s"))
        self.add_component('variable', DifferentialVariable(0.1, "m/s"))
        self.add_component('dummy', Variable(0.0, "m/s"))

    def declare_equations(self):
        eq2 = self['der_variable'].symbol - self['p'].symbol * self['dummy'].symbol
        return [eq2]


class InnerODE_2(Model):
    """`der_variable = variable` -> exponential growth."""

    def __init__(self):
        super().__init__()

    def declare_components(self):
        self.add_component('variable', DifferentialVariable(0.1, "m/s"))

    def declare_equations(self):
        eq2 = self['der_variable'].symbol - self['variable'].symbol
        return [eq2]


class SimpleODE(Model):
    """Composite model linking `InnerODE_1.dummy` to `InnerODE_2.variable`."""

    def declare_components(self):
        self.add_component('inner_ode1', InnerODE_1())
        self.add_component('inner_ode2', InnerODE_2())

    def declare_equations(self):
        eq1 = self['inner_ode1']['dummy'].symbol - self['inner_ode2']['variable'].symbol
        return [eq1]


class IntegrationTest(Model):
    """
    Decoupled ODEs added purely to validate the time integrator. Each of the
    differential variables below has a closed-form solution that we compare
    against after the simulation loop.

      1. Exponential decay:
             dy/dt = -y,                 y(0) = 1
             analytical:  y(t) = exp(-t)

      2. Harmonic oscillator (omega defaults to 2*pi -> period of 1 s):
             dy/dt = z
             dz/dt = -omega**2 * y,      y(0) = 1, z(0) = 0
             analytical:  y(t) =  cos(omega*t)
                          z(t) = -omega * sin(omega*t)
    """

    def __init__(self, omega=2 * np.pi):
        self.omega_value = omega
        super().__init__()

    def declare_components(self):
        # exponential decay
        self.add_component('y_decay', DifferentialVariable(1.0, None))

        # harmonic oscillator
        self.add_component('omega', Parameter(self.omega_value, "1/s"))
        self.add_component('y_osc', DifferentialVariable(1.0, None))
        self.add_component('z_osc', DifferentialVariable(0.0, None))

    def declare_equations(self):
        eq_decay = self['der_y_decay'].symbol + self['y_decay'].symbol
        eq_osc_y = self['der_y_osc'].symbol - self['z_osc'].symbol
        eq_osc_z = self['der_z_osc'].symbol + self['omega'].symbol ** 2 * self['y_osc'].symbol
        return [eq_decay, eq_osc_y, eq_osc_z]
