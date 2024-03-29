# Ongoing Work

## TODOs

- Add advanced continuation methods
- Streamline problem structe
- Add annotations to solution class
- Add finite difference necesary conditions
- Add interactive visualization
- Add "Active monitoring"
- Add custom BVP solver
- Add direct solver

## Ideas

- Explore using "post-processers" on problems for solution transformations/computations (e.g. true control, aux values, cost)
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
- Consider using a Protocol to avoid typing error with Scipy BVP solution (Addressed with Generic)
- Add support for multiple checks when using Picky
- Assign tuples to ndarrays to make output consistent and elimante instances of "double complilation"
- Analyze trade-off of JIT complilation and runtime as function of problem size
