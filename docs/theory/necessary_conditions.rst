Derivation of Necessary Conditions
##################################

More information can be found in [1]_, [2]_.

A single stage optimal control problem (OCP) seeks the trajectory which minimizes the cost functional :math:`J` with respect to a control :math:`\boldsymbol{u}{\left(t\right)}`.
Here, we define the **independent variable** (a.k.a **time**) :math:`t` and continuously variable **state** :math:`\boldsymbol{x}{\left(t\right)}`.
The control vector :math:`\boldsymbol{u}{\left(t\right)}` is assumed piecewise-continuous.
For breviety, the time dependency on the states and controls will be assumed and :math:`\left(t\right)` will be ommited.
The subscript :math:`_0` denotes value evaluated at the initial time :math:`t_0`, and likewise, :math:`_f` denotes evaulation at final time :math:`t_f`.

The cost functional is made of initial :math:`\phi_0`, terminal :math:`\phi_f`, and path :math:`L` components:

.. math::
    \min_{u} J = J = \phi_0{\left(t_0, \boldsymbol{x}_0\right)} + \phi_f{\left(t_f, \boldsymbol{x}_f\right)} + \int_{t_0}^{t_f}{L{\left(t, \boldsymbol{x}, \boldsymbol{u}\right)} \, dt}

The trajectory is subject to the state dynamics

.. math::
    \dot{\boldsymbol{x}} = \boldsymbol{f}{\left(t, \boldsymbol{x}, \boldsymbol{u}\right)}

and boundary conditions:

.. math::
    \boldsymbol{\Psi}_0{\left(t_0, \boldsymbol{x}_0\right)} = 0 \\
    \boldsymbol{\Psi}_f{\left(t_f, \boldsymbol{x}_f\right)} = 0

To enforce the constraints, we adjoined them to the cost functional with Lagrange multipliers :math:`\boldsymbol{\nu}_0`, :math:`\boldsymbol{\nu}_f`, and :math:`\boldsymbol{\lambda}{\left(t\right)}`:

.. math::
    \bar{J} = \phi_0{\left(t_0, \boldsymbol{x}_0\right)} + \boldsymbol{\nu}_0^T \boldsymbol{\Psi}_0{\left(t_0, \boldsymbol{x}_0\right)} + \phi_f{\left(t_f, \boldsymbol{x}_f\right)} + \boldsymbol{\nu}_f^T \boldsymbol{\Psi}_f{\left(t_f, \boldsymbol{x}_f\right)} \\
    + \int_{t_0}^{t_f}{L{\left(t, \boldsymbol{x}, \boldsymbol{u}\right)} + \boldsymbol{\lambda}^T{\left[\boldsymbol{f}{\left(t, \boldsymbol{x}, \boldsymbol{u}\right)} - \dot{\boldsymbol{x}}\right]}\, dt}

The Lagrange multipliers that adjoin the state dynamics :math:`\boldsymbol{\lambda}{\left(t\right)}` are referred to as **costates**.
Grouping the terms and removing function arguments for clearity yields

.. math::
    \bar{J} = \left[\phi_0 + \boldsymbol{\nu}_0^T \boldsymbol{\Psi}_0\right]_{t=t_0} + \left[\phi_f + \boldsymbol{\nu}_f^T \boldsymbol{\Psi}_f\right]_{t=t_f} + \int_{t_0}^{t_f}{L{\left(t, \boldsymbol{x}, \boldsymbol{u}\right)} + \boldsymbol{\lambda}^T{\left[\boldsymbol{f} - \dot{\boldsymbol{x}}\right]}\, dt}

For conciseness, we define :math:`\Phi_0 = \phi_0 + \boldsymbol{\nu}_0^T \boldsymbol{\Psi}_0`, :math:`\Phi_f = \phi_f + \boldsymbol{\nu}_f^T \boldsymbol{\Psi}_f`, and **Hamiltonian** :math:`H = L + \boldsymbol{\lambda}^T\boldsymbol{f}`:

.. math::
    \bar{J} = \Phi_0 + \Phi_f + \int_{t_0}^{t_f}{\left[H - \boldsymbol{\lambda}^T\dot{\boldsymbol{x}}\right] \, dt}

The primary necessary condition of optimality is that the first differential of the adjoined cost functional is zero:

.. math::
    \delta \bar{J} = \left[\frac{\partial \Phi_0}{\partial t} dt + \frac{\partial \Phi_0}{\partial\boldsymbol{x}} d \boldsymbol{x} \right]_{t=t_0} + \left[\frac{\partial \Phi_f}{\partial t} dt + \frac{\partial \Phi_f}{\partial \boldsymbol{x}} d \boldsymbol{x} \right]_{t=t_f} \\
    + \left[L \, dt\right]_{t_0}^{t_f} + \int_{t_0}^{t_f}{\left[\frac{\partial H}{\partial \boldsymbol{x}} \delta \boldsymbol{x} + \frac{\partial H}{\partial \boldsymbol{u}} \delta \boldsymbol{u} - \boldsymbol{\lambda}^T \delta\dot{\boldsymbol{x}} \right] \, dt} = 0

By distributing :math:`\left[L \, dt\right]_{t_0}^{t_f}`:

.. math::
    \delta \bar{J} = \left[\left(\frac{\partial \Phi_0}{\partial t} - L\right) dt + \frac{\partial \Phi_0}{\partial\boldsymbol{x}} d \boldsymbol{x} \right]_{t=t_0} + \left[\left(\frac{\partial \Phi_f}{\partial t} + L\right) dt + \frac{\partial \Phi_f}{\partial \boldsymbol{x}} d \boldsymbol{x} \right]_{t=t_f} \\
    + \int_{t_0}^{t_f}{\left[\frac{\partial H}{\partial \boldsymbol{x}} \delta \boldsymbol{x} + \frac{\partial H}{\partial \boldsymbol{u}} \delta \boldsymbol{u} - \boldsymbol{\lambda}^T \delta\dot{\boldsymbol{x}} \right] \, dt} = 0

We remove the dependency on :math:`\delta \dot{\boldsymbol{x}}` via integration by parts:

.. math::
    \int{u \, dv} = uv - \int{v \, du}

.. math::
    \Rightarrow \int_{t_0}^{t_f} -\boldsymbol{\lambda}^T \delta \dot{\boldsymbol{x}} \, dt = \left[-\boldsymbol{\lambda}^T \delta \boldsymbol{x}\right]_{t_0}^{t_f} + \int_{t_0}^{t_f}{\dot{\boldsymbol{\lambda}}^T \delta \boldsymbol{x} \, dt}

Therefore:

.. math::
    \delta \bar{J} = \left[\left(\frac{\partial \Phi_0}{\partial t} - L\right) dt + \frac{\partial \Phi_0}{\partial\boldsymbol{x}} d \boldsymbol{x} \right]_{t=t_0} + \left[\left(\frac{\partial \Phi_f}{\partial t} + L\right) dt + \frac{\partial \Phi_f}{\partial \boldsymbol{x}} d \boldsymbol{x} \right]_{t=t_f} \\
                   + \left[-\boldsymbol{\lambda}^T \delta \boldsymbol{x}\right]_{t_0}^{t_f} + \int_{t_0}^{t_f}{\left[\left(\frac{\partial H}{\partial \boldsymbol{x}} + \dot{\boldsymbol{\lambda}}^T\right) \delta \boldsymbol{x} + \frac{\partial H}{\partial \boldsymbol{u}} \delta \boldsymbol{u} \right] \, dt} = 0

Noting :math:`\delta \boldsymbol{x} = d \boldsymbol{x} - \dot{\boldsymbol{x}} \, dt`:

.. math::
    \delta \bar{J} = \left[\left(\frac{\partial \Phi_0}{\partial t} - L\right) dt + \frac{\partial \Phi_0}{\partial\boldsymbol{x}} d \boldsymbol{x} \right]_{t=t_0} + \left[\left(\frac{\partial \Phi_f}{\partial t} + L\right) dt + \frac{\partial \Phi_f}{\partial \boldsymbol{x}} d \boldsymbol{x} \right]_{t=t_f} \\
                   + \left[-\boldsymbol{\lambda}^T \left(d \boldsymbol{x} - \dot{\boldsymbol{x}} \, dt\right)\right]_{t_0}^{t_f} + \int_{t_0}^{t_f}{\left[\left(\frac{\partial H}{\partial \boldsymbol{x}} + \dot{\boldsymbol{\lambda}}^T\right) \delta \boldsymbol{x} + \frac{\partial H}{\partial \boldsymbol{u}} \delta \boldsymbol{u} \right] \, dt} = 0

Combining terms:

.. math::
    \delta \bar{J} = \left[\left(\frac{\partial \Phi_0}{\partial t} - L - \boldsymbol{\lambda}^T \dot{\boldsymbol{x}}\right) dt + \left(\frac{\partial \Phi_0}{\partial\boldsymbol{x}} + \boldsymbol{\lambda}^T \right) d \boldsymbol{x} \right]_{t=t_0} \\
                   + \left[\left(\frac{\partial \Phi_f}{\partial t} + L + \boldsymbol{\lambda}^T \dot{\boldsymbol{x}}\right) dt + \left(\frac{\partial \Phi_f}{\partial\boldsymbol{x}} - \boldsymbol{\lambda}^T \right) d \boldsymbol{x} \right]_{t=t_f} \\
                   + \int_{t_0}^{t_f}{\left[\left(\frac{\partial H}{\partial \boldsymbol{x}} + \dot{\boldsymbol{\lambda}}^T\right) \delta \boldsymbol{x} + \frac{\partial H}{\partial \boldsymbol{u}} \delta \boldsymbol{u} \right] \, dt} = 0

Expressing in terms of the Hamiltonian:

.. math::
    \delta \bar{J} = \left[\left(\frac{\partial \Phi_0}{\partial t} - H \right) dt + \left(\frac{\partial \Phi_0}{\partial\boldsymbol{x}} + \boldsymbol{\lambda}^T \right) d \boldsymbol{x} \right]_{t=t_0} \\
                   + \left[\left(\frac{\partial \Phi_f}{\partial t} + H \right) dt + \left(\frac{\partial \Phi_f}{\partial\boldsymbol{x}} - \boldsymbol{\lambda}^T \right) d \boldsymbol{x} \right]_{t=t_f} \\
                   + \int_{t_0}^{t_f}{\left[\left(\frac{\partial H}{\partial \boldsymbol{x}} + \dot{\boldsymbol{\lambda}}^T\right) \delta \boldsymbol{x} + \frac{\partial H}{\partial \boldsymbol{u}} \delta \boldsymbol{u} \right] \, dt} = 0

The desired first-order necesary coditions are obtained by setting all non-differential terms to zero.
This contains dynamic equations for the costates:

.. math::
    \dot{\boldsymbol{\lambda}} = - \left(\frac{\partial H}{\partial \boldsymbol{x}} \right)^T

A system of algebraic equations which constitute the optimal control law:

.. math::
    0 = \frac{\partial H}{\partial \boldsymbol{u}}

And additional boundary conditions:

.. math::
    0 = \frac{\partial \Phi_0}{\partial t} - H_0 = \frac{\partial \phi_0}{\partial t} + \boldsymbol{\nu}_0^T \frac{\partial \boldsymbol{\Psi}_0}{\partial t} - H_0 \\
    0 = \frac{\partial \Phi_f}{\partial t} + H_f = \frac{\partial \phi_f}{\partial t} + \boldsymbol{\nu}_f^T \frac{\partial \boldsymbol{\Psi}_f}{\partial t} + H_f \\
    0 = \frac{\partial \Phi_0}{\partial \boldsymbol{x}} + \boldsymbol{\lambda}_0^T = \frac{\partial \phi_0}{\partial \boldsymbol{x}} + \boldsymbol{\nu}_0^T \frac{\partial \boldsymbol{\Psi}_0}{\partial \boldsymbol{x}} + \boldsymbol{\lambda}_0^T \\
    0 = \frac{\partial \Phi_f}{\partial \boldsymbol{x}} - \boldsymbol{\lambda}_f^T = \frac{\partial \phi_f}{\partial \boldsymbol{x}} + \boldsymbol{\nu}_f^T \frac{\partial \boldsymbol{\Psi}_f}{\partial \boldsymbol{x}} - \boldsymbol{\lambda}_f^T \\

.. [1]  Bryson, A. E., and Ho, Y.-C. Applied Optimal Control: Optimization, Estimation, and Control. Hemisphere Publishing Corporation, Washinngton, DC, 1975.
.. [2]  Longuski, J. M., Guzman, J. J., and Prussing, J. E. Optimal Control with Aerospace Applications. Springer New York, New York, NY, 2014.
