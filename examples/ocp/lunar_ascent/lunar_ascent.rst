Lunar Ascent Problem
====================

This problem is taken from Applied Optimal Control by Bryson and Ho [1]_.
It can be found as Problem 13 in Section 2.7 in the 1975 edition.

It solves for the minimum-time orbit injection assuming no drag, a constant thrust-acceleration, and constant gravity field.

Variables
---------

.. math::
    :name: Independent

    t &= \text{time [s]}

.. math::
    :name: States

    x &= \text{dowrange [ft]} \\
    y &= \text{altitude [ft]} \\
    v_x &= \text{horizontal velocity [ft/s]} \\
    v_y &= \text{vertical velocity [ft/s]} \\

.. math::
    :name: Control

    β &= \text{thrust angle [rad]} \\

.. math::
    :name: Constants

    a &= \text{thrust-acceration} \\
    g &= \text{gravity} \\
    r_m &= \text{radius of moon} \\


.. [1] Bryson, A. E., and Ho, Y.-C. Applied Optimal Control: Optimization, Estimation, and Control. Hemisphere Publishing Corporation, Washinngton, DC, 1975.
