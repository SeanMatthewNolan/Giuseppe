Giuseppe
********

.. automodule:: giuseppe
   :members:

Continuation
************

.. automodule:: giuseppe.continuation
   :members:

Solution Set
############

.. autoclass:: giuseppe.continuation.solution_set.SolutionSet
   :members:
   :special-members: __init__
   :undoc-members:

Continuation Handler
####################

.. autoclass:: giuseppe.continuation.handler.ContinuationHandler
   :members:
   :special-members: __init__
   :undoc-members:

Methods
#######

.. automodule:: giuseppe.continuation.methods
   :members:
   :undoc-members:

Abstract
========

.. autoclass:: giuseppe.continuation.methods.abstract.ContinuationSeries
   :members:
   :special-members: __init__, __iter__
   :undoc-members:

Linear
======

.. autoclass:: giuseppe.continuation.methods.linear.LinearSeries
   :members:
   :special-members: __init__
   :undoc-members:

.. autoclass:: giuseppe.continuation.methods.linear.BisectionLinearSeries
   :members:
   :special-members: __init__
   :undoc-members:

Logarithmic
===========

.. autoclass:: giuseppe.continuation.methods.logarithmic.LogarithmicSeries
   :members:
   :special-members: __init__
   :undoc-members:

.. autoclass:: giuseppe.continuation.methods.logarithmic.BisectionLogarithmicSeries
   :members:
   :special-members: __init__
   :undoc-members:

Guess Generators
****************

.. automodule:: giuseppe.guess_generators
   :members:

Constant
########

.. autofunction:: giuseppe.guess_generators.constant.generate_constant_guess

Input/Output
************

.. automodule:: giuseppe.io
   :members:

Numeric Solvers
***************

.. automodule:: giuseppe.numeric_solvers
   :members:

BVP Solvers
###########

.. automodule:: giuseppe.numeric_solvers.bvp
   :members:

SciPy
=====

.. autoclass:: giuseppe.numeric_solvers.bvp.scipy.ScipySolveBVP
   :members:
   :special-members: __init__
   :undoc-members:

Problem Classes
***************

.. automodule:: giuseppe.problems
   :members:

Utilities
*********

.. automodule:: giuseppe.utils
   :members:
