Predator-Prey Problem
====================

This problem is taken from Optimal Control with Aerospace Applications by Longuski, Guzmán, and Prussing [1]_.
It can be found as Example 2.7 in Section 2.4 in the 2014 edition.

It uses the Lotka-Volterra model to simulate a predator-prey relationship into which pesticide is introduced.
The imagined scenario is on of a farmer who is raising crops (prey) which are damaged by insects (predators).
The farmer can kill off the insects with pesticide, but the pesticide costs money.
The farmer wants to maximize profits over a given period.

Variables
---------

.. math::
    :name: Independent

    t &= \text{time [s]}

.. math::
    :name: States

    x_1 &= \text{number of prey [scaled, dimensionless]} \\
    x_2 &= \text{number of predators [scaled, dimensionless]} \\

.. math::
    :name: Control

    u &= \text{rate of pesticide introduction} \\

.. math::
    :name: Constants

    a &= \text{cost of pesticide relative to crops} \\
    l &= \text{pesticide effectiveness} \\
    k &= \text{death rate of predators}

.. math::
    :name: Dynamics

    \dot{x}_1 &= x_1 - x_1 x_2 \\
    \dot{x}_2 &= x_1 x_2

.. math::
    :name: Cost

    J &= -x_{1,f} + \int_0^{t_f} au dt

.. math::
    :name: Boundary Conditions

    \mathbf{\Psi}_0 &= \left[t, x_1 - x_{1,0}, x_2 - x_{2,0}\right]^T \\
    \mathbf{\Psi}_f &= \left[t - t_f\right]^T

.. [1] Longuski, J. M., Guzmán, J. J., and Prussing, J. E. Optimal Control with Aerospace Applications. Springer New York, New York, NY, 2014.
