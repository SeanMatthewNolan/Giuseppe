---
title: Giuseppe Abbreviations and Definitions
author: Sean Matthew Nolan
date: 3/11/22
---

# Abbreviations

- **BVP**: Boundary Value Problem
- **Comp**: Compiled, for problems and functions with have been turned in regular Python functions and the JIT compiled with Numba
- **Dual**: Dualized, for problems with dual information (see below for *dual* definition)
- **IO**: Input/Output
- **OCP**: Optimal Control Problem
- **src**: Source
- **Sym**: Symbolic, in code refers to objects which use SymPy to represent symbolic expressions 


# Definitions

The below definitions cover how these words are used in Giuseppe:

- **adjoint**: (AKA **dual**, **Lagrange multiplier**) variable needed to **adjoin** **constraints** to **cost** functional, measure of sensitivity of the **objective** to the **constraint**
- **augment**: see **dualize**
- **augmented**: expression, **cost** component, with constraints **adjoined**  
- **constant**: known value (non-**dynamic**) set by user, used to specify **continuation** steps
- **constraint**: equation that must be satisfied for a feasible **solution** to the **OCP**
- **control**: unknown variables free to change over the **trajectory** to minimize the **cost** and satisfy **constraints**, not **dynamically** constrained directly
- **cost**: functional that the **OCP** seeks to minimize while satisfying **constraints**
- **costate**: **dynamic** **adjoint** variable (**dual** to a **state**) 
- **dual**: see **adjoint**
- **dualize**: (AKA **augment**, **adjoin**) process of adding the constraint information to the **cost** by using **Lagrange multipliers** 
- **dynamic**: changes w.r.t **independent** variable
- **dynamics**: (AKA *dynamic equations*, *equations of motion*) equations of first derivative w.r.t **independent** variable
- **Hamiltonian**: path cost with adjoined with path equality constraints and the inner product of the **costates** and **state** **dynamics**
- **independent**: variable by which **dynamic** variables change, usually *time*, *t* 
- **initial**: evaluated at beginning point of **trajectory**
- **Lagrange multiplier**: see **adjoint**
- **parameter**: non-**dynamic** (constant) unknown
- **path**: evaluated/integrated over the **trajectory**
- **solution**: **trajectory** that locally minimizes the **cost** while satisfying the constraints (within tolerances)
- **state**: **dynamic** unknown in problems (constrained by specified **dynamics**)
- **terminal**: evaluated at end point of **trajectory**
- **trajectory**: numerical data for the variables of interest over the interval of interests 

