---
title: Giuseppe TODOs and Ideas
author: Sean Matthew Nolan
date: 3/14/22
---

# TODOs

- Add compilation classes
- Add numerical solvers
- Add continuation handlers
- Add guess generators
- Add parameter support
- Add inequality constraints 

# Ideas

- Use SciPy's PPoly to represent some solutions
- Explore "custom functions" via making function to evaluate `Derivative(*, *)`
- Explore SymPy's `Piecewise` for univariate tables
- Explore SymPy's `xreplace` for quantities
- Explore Numba's `cfunc` for use with C++ solvers
- Explore split vs. combined BCs
- Make typing more explicit with Generic types
- Pre-compile Jacobians (issues with differential controls)
- Implement "annotate" method to return solutions with names from SymBVP
    - Consider a "metadata" class shared by all problems
- Consider using a Protocol to avoid typing error with Scipy BVP solution
- Add support for multiple checks when using Picky
