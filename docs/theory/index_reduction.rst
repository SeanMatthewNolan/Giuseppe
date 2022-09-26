Index Reduction for Differential Control Law
############################################

More information can be found in [1]_, [2]_.

A system of algebraic equations which constitute the optimal control law:

.. math::
    0 = \frac{\partial H}{\partial \boldsymbol{u}}

The need to solve the above for an explicit, analytic function for the optimal control law has long been considered an obstacle to using indirect methods [3]_, [4]_.
For many problems, especially ones with complex dynamics and constraints, finding such a function is not possible.
The first instinct to solve the problem in such a case may be to use a numerical root solving technique, such as Newton's method, to find the control law.
Unfortunately, such an approach is computationally expensive and prone to non-convergence.

Index-reduction is an approach that has been proved successful for solving these complex problems.
It works by finding an expression for the the derivative of the control (:math:`\dot{u}`)
The first step is to take the time derivative of :math:`\frac{\partial H}{\partial \boldsymbol{u}}` taking the chain rule into account.

.. math::
    0 &= \frac{d}{dt} \frac{\partial H}{\partial\boldsymbol{u}} \\
      &= \frac{\partial^{2} H}{\partial t \partial \boldsymbol{u}} + \frac{\partial^{2}H}{\partial\boldsymbol{x}\partial\boldsymbol{u}} \frac{d\boldsymbol{x}}{dt} + \frac{\partial^{2}H}{\partial\boldsymbol{\lambda}\partial\boldsymbol{u}} \frac{d\boldsymbol{\lambda}}{dt} + \frac{\partial^{2}H}{\partial\boldsymbol{u}^2} \frac{d\boldsymbol{u}}{dt} \\
      &= {H}_{\boldsymbol{u}t} + {H}_{\boldsymbol{ux}}\dot{\boldsymbol{x}} + {H}_{\boldsymbol{u\lambda}}\dot{\boldsymbol{\lambda}} + {H}_{\boldsymbol{uu}}\dot{\boldsymbol{u}}

The above may then be arranged for explicit expression for :math:`\dot{\boldsymbol{u}}` assuming :math:`{H}_{\boldsymbol{uu}}` is invertible:

.. math::
    \dot{\boldsymbol{u}} = -{H}_{\boldsymbol{uu}}^{-1} \left({H}_{\boldsymbol{u}t} + {H}_{\boldsymbol{ux}}\dot{\boldsymbol{x}} + {H}_{\boldsymbol{u\lambda}}\dot{\boldsymbol{\lambda}}\right)

With the above, the control can then be solved alongside the other differential variables (i.e. states and costates).
The original system of equations, :math:`{H}_{\boldsymbol{u}} = 0`, then serves as boundary conditions placed either at the beginning or the end of the problem.
When numerically solving for :math:`\dot{\boldsymbol{u}}`, it usually more efficient to solve the system of equations in the form:

.. math::
    {H}_{\boldsymbol{uu}}\dot{\boldsymbol{u}} = {H}_{\boldsymbol{u}t} + {H}_{\boldsymbol{ux}}\dot{\boldsymbol{x}} + {H}_{\boldsymbol{u\lambda}}\dot{\boldsymbol{\lambda}}

Depending on the implementation, it may helpful to make the following substitutions:

.. math::
   \dot{\boldsymbol{x}} &= \boldsymbol{f}{\left(t, \boldsymbol{x}, \boldsymbol{u}\right)} = -{H}_{\boldsymbol{\lambda}}^T \\
   \dot{\boldsymbol{\lambda}} &= -{H}_{\boldsymbol{x}}^T \\
   {H}_{\boldsymbol{u \lambda}} &= -\frac{\partial}{\partial \boldsymbol{u}}\left[\boldsymbol{f}{\left(t, \boldsymbol{x}, \boldsymbol{u}\right)}\right]^T \\

.. [1] Antony, T., and Grant, M. J. Path Constraint Regularization in Optimal Control Problems Using Saturation Functions. Presented at the 2018 AIAA Atmospheric Flight Mechanics Conference, Kissimmee, Florida, 2018.
.. [2] Führer, C., and Leimkuhler, B. J. “Numerical Solution of Differential-Algebraic Equations for Constrained Mechanical Motion.” Numerische Mathematik, Vol. 59, No. 1, 1991, pp. 55–69. https://doi.org/10.1007/BF01385770.
.. [3] Betts, J. T. “Survey of Numerical Methods for Trajectory Optimization.” Journal of Guidance, Control, and Dynamics, Vol. 21, No. 2, 1998, pp. 193–207. https://doi.org/10.2514/2.4231.
.. [4] Shirazi, A., Ceberio, J., and Lozano, J. A. “Spacecraft Trajectory Optimization: A Review of Models, Objectives, Approaches and Solutions.” Progress in Aerospace Sciences, Vol. 102, 2018, pp. 76–98. https://doi.org/10.1016/j.paerosci.2018.07.007.
