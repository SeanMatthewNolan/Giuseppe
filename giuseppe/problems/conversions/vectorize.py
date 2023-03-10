from typing import Union, cast
import copy

import numpy as np

from ..protocols import Problem, VectorizedBVP, VectorizedOCP, VectorizedAdjoints, VectorizedDual,\
    AlgebraicControlHandler, DifferentialControlHandler,  VectorizedAlgebraicControlHandler,\
    VectorizedDifferentialControlHandler
from ...utils.compilation import check_if_can_jit_compile, jit_compile
from ...utils.typing import NumbaArray, NumbaMatrix


def vectorize(
        input_prob: Problem, use_jit_compile=True
) -> Union[VectorizedBVP, VectorizedOCP, VectorizedAdjoints, VectorizedDual]:

    if check_if_can_jit_compile(use_jit_compile, input_prob):
        return _jit_vectorized(input_prob)
    else:
        prob = copy.deepcopy(input_prob)

        if hasattr(prob, 'use_jit_compile'):
            prob.use_jit_compile = False

        if input_prob.prob_class in ['bvp', 'ocp', 'dual']:

            _compute_dynamics = input_prob.compute_dynamics

            if input_prob.prob_class == 'bvp':
                def _compute_dynamics_vectorized(
                        independent: np.ndarray, states: np.ndarray, parameters: np.ndarray, constants: np.ndarray
                ) -> np.ndarray:
                    return np.vstack(
                            tuple(_compute_dynamics(ti, xi, parameters, constants)
                                  for ti, xi in zip(independent, states.T))).T

                prob.compute_dynamics_vectorized = _compute_dynamics_vectorized

                prob = cast(VectorizedBVP, prob)

            else:
                _compute_path_cost = prob.compute_path_cost

                def _compute_dynamics_vectorized(
                        independent: np.ndarray, states: np.ndarray, controls: np.ndarray, parameters: np.ndarray,
                        constants: np.ndarray
                ) -> np.ndarray:
                    return np.vstack(
                            tuple(_compute_dynamics(ti, xi, ui, parameters, constants)
                                  for ti, xi, ui in zip(independent, states.T, controls.T))).T

                prob.compute_dynamics_vectorized = _compute_dynamics_vectorized

                def _compute_path_cost_vectorized(
                        independent: np.ndarray, states: np.ndarray, controls: np.ndarray, parameters: np.ndarray,
                        constants: np.ndarray
                ) -> np.ndarray:
                    return np.array(
                            tuple(_compute_path_cost(ti, xi, ui, parameters, constants)
                                  for ti, xi, ui in zip(independent, states.T, controls.T)))

                prob.compute_path_cost_vectorized = _compute_path_cost_vectorized

                prob.compute_cost = VectorizedOCP.compute_cost

                prob = cast(VectorizedOCP, prob)

        if input_prob.prob_class in ['adjoints', 'dual']:
            _compute_costate_dynamics = input_prob.compute_costate_dynamics
            _compute_hamiltonian = input_prob.compute_hamiltonian
            _compute_control_law = input_prob.compute_control_law

            def _compute_costates_dynamics_vectorized(
                    independent: np.ndarray, states: np.ndarray, costates: np.ndarray, controls: np.ndarray,
                    parameters: np.ndarray, constants: np.ndarray
            ) -> np.ndarray:
                return np.vstack(
                        tuple(_compute_costate_dynamics(ti, xi, lami, ui, parameters, constants)
                              for ti, xi, lami, ui in zip(independent, states.T, costates.T, controls.T))).T

            prob.compute_costate_dynamics_vectorized = _compute_costates_dynamics_vectorized

            def _compute_hamiltonian_vectorized(
                    independent: np.ndarray, states: np.ndarray, costates: np.ndarray, controls: np.ndarray,
                    parameters: np.ndarray, constants: np.ndarray
            ) -> np.ndarray:
                return np.array(
                        tuple(_compute_hamiltonian(ti, xi, lami, ui, parameters, constants)
                              for ti, xi, lami, ui in zip(independent, states.T, costates.T, controls.T)))

            prob.compute_hamiltonian_vectorized = _compute_hamiltonian_vectorized

            def _compute_control_law_vectorized(
                    independent: np.ndarray, states: np.ndarray, costates: np.ndarray, controls: np.ndarray,
                    parameters: np.ndarray, constants: np.ndarray
            ) -> np.ndarray:
                return np.vstack(
                        tuple(_compute_control_law(ti, xi, lami, ui, parameters, constants)
                              for ti, xi, lami, ui in zip(independent, states.T, costates.T, controls.T))).T

            prob.compute_control_law_vectorized = _compute_control_law_vectorized

            prob = cast(VectorizedAdjoints, prob)

        if hasattr(prob, 'control_handler') and prob.control_handler is not None:
            if isinstance(prob.control_handler, AlgebraicControlHandler):
                _compute_control = prob.control_handler.compute_control

                def _compute_control_vectorized(
                        independent: np.ndarray, states: np.ndarray, costates: np.ndarray,
                        parameters: np.ndarray, constants: np.ndarray
                ) -> np.ndarray:
                    return np.vstack(
                            tuple(_compute_control(ti, xi, lami, parameters, constants)
                                  for ti, xi, lami in zip(independent, states.T, costates.T))).T

                prob.control_handler.compute_control_vectorized = _compute_control_vectorized

            elif isinstance(prob.control_handler, DifferentialControlHandler):
                _compute_control_dynamics = prob.control_handler.compute_control_dynamics

                def _compute_control_dynamics_vectorized(
                        independent: np.ndarray, states: np.ndarray, costates: np.ndarray, controls: np.ndarray,
                        parameters: np.ndarray, constants: np.ndarray
                ) -> np.ndarray:
                    return np.vstack(
                            tuple(_compute_control_dynamics(ti, xi, lami, ui, parameters, constants)
                                  for ti, xi, lami, ui in zip(independent, states.T, costates.T, controls.T))).T

                prob.control_handler.compute_control_dynamics_vectorized = _compute_control_dynamics_vectorized

        return prob


def _jit_vectorized(input_prob: Problem) -> Union[VectorizedBVP, VectorizedOCP, VectorizedAdjoints, VectorizedDual]:
    prob = copy.deepcopy(input_prob)

    if input_prob.prob_class in ['bvp', 'ocp', 'dual']:
        _compute_dynamics = input_prob.compute_dynamics

        if input_prob.prob_class == 'bvp':
            def _compute_dynamics_vectorized(
                    independent: np.ndarray, states: np.ndarray, parameters: np.ndarray, constants: np.ndarray
            ) -> np.ndarray:
                x_dot = np.empty_like(states)  # Need to pre-allocate for Numba
                for idx, (ti, xi) in enumerate(zip(independent, states.T)):
                    x_dot[:, idx] = _compute_dynamics(ti, xi, parameters, constants)

                return x_dot

            prob.compute_dynamics_vectorized = jit_compile(
                    _compute_dynamics_vectorized,
                    (NumbaArray, NumbaMatrix, NumbaArray, NumbaArray)
            )

            prob = cast(VectorizedBVP, prob)

        else:
            _compute_path_cost = prob.compute_path_cost

            def _compute_dynamics_vectorized(
                    independent: np.ndarray, states: np.ndarray, controls: np.ndarray, parameters: np.ndarray,
                    constants: np.ndarray
            ) -> np.ndarray:

                x_dot = np.empty_like(states)  # Need to pre-allocate for Numba
                for idx, (ti, xi, ui) in enumerate(zip(independent, states.T, controls.T)):
                    x_dot[:, idx] = _compute_dynamics(ti, xi, ui, parameters, constants)

                return x_dot

            prob.compute_dynamics_vectorized = jit_compile(
                    _compute_dynamics_vectorized,
                    (NumbaArray, NumbaMatrix, NumbaMatrix, NumbaArray, NumbaArray)
            )

            def _compute_path_cost_vectorized(
                    independent: np.ndarray, states: np.ndarray, controls: np.ndarray, parameters: np.ndarray,
                    constants: np.ndarray
            ) -> np.ndarray:

                lagrangians = np.empty_like(independent)
                for idx, (ti, xi, ui) in enumerate(zip(independent, states.T, controls.T)):
                    lagrangians[idx] = _compute_path_cost(ti, xi, ui, parameters, constants)

                return lagrangians

            prob.compute_path_cost_vectorized = jit_compile(
                    _compute_path_cost_vectorized,
                    (NumbaArray, NumbaMatrix, NumbaMatrix, NumbaArray, NumbaArray)
            )

            prob.compute_cost = VectorizedOCP.compute_cost

            prob = cast(VectorizedOCP, prob)

    if input_prob.prob_class in ['adjoints', 'dual']:
        _compute_costate_dynamics = input_prob.compute_costate_dynamics
        _compute_hamiltonian = input_prob.compute_hamiltonian
        _compute_control_law = input_prob.compute_control_law

        def _compute_costate_dynamics_vectorized(
                independent: np.ndarray, states: np.ndarray, costates: np.ndarray, controls: np.ndarray,
                parameters: np.ndarray, constants: np.ndarray
        ) -> np.ndarray:

            lam_dot = np.empty_like(costates)  # Need to pre-allocate for Numba
            for idx, (ti, xi, lami, ui) in enumerate(zip(independent, states.T, costates.T, controls.T)):
                lam_dot[:, idx] = _compute_costate_dynamics(ti, xi, lami, ui, parameters, constants)

            return lam_dot

        prob.compute_costate_dynamics_vectorized = jit_compile(
                _compute_costate_dynamics_vectorized,
                (NumbaArray, NumbaMatrix, NumbaMatrix, NumbaMatrix, NumbaArray, NumbaArray)
        )

        def _compute_hamiltonian_vectorized(
                independent: np.ndarray, states: np.ndarray, costates: np.ndarray, controls: np.ndarray,
                parameters: np.ndarray, constants: np.ndarray
        ) -> np.ndarray:

            hamiltonians = np.empty_like(independent)
            for idx, (ti, xi, lami, ui) in enumerate(zip(independent, states.T, costates.T, controls.T)):
                hamiltonians[idx] = _compute_hamiltonian(ti, xi, lami, ui, parameters, constants)

            return hamiltonians

        prob.compute_hamiltonian_vectorized = jit_compile(
                _compute_hamiltonian_vectorized,
                (NumbaArray, NumbaMatrix, NumbaMatrix, NumbaMatrix, NumbaArray, NumbaArray)
        )

        def _compute_control_law_vectorized(
                independent: np.ndarray, states: np.ndarray, costates: np.ndarray, controls: np.ndarray,
                parameters: np.ndarray, constants: np.ndarray
        ) -> np.ndarray:
            dh_du = np.empty_like(controls)  # Need to pre-allocate for Numba
            for idx, (ti, xi, lami, ui) in enumerate(zip(independent, states.T, costates.T, controls.T)):
                dh_du[:, idx] = _compute_control_law(ti, xi, lami, ui, parameters, constants)

            return dh_du

        prob.compute_control_law_vectorized = jit_compile(
                _compute_control_law_vectorized,
                (NumbaArray, NumbaMatrix, NumbaMatrix, NumbaMatrix, NumbaArray, NumbaArray)
        )

        prob = cast(VectorizedAdjoints, prob)

    if hasattr(prob, 'control_handler') and prob.control_handler is not None:
        if isinstance(prob.control_handler, AlgebraicControlHandler):
            _compute_control = prob.control_handler.compute_control
            _num_controls = prob.num_controls

            def _compute_control_vectorized(
                    independent: np.ndarray, states: np.ndarray, costates: np.ndarray,
                    parameters: np.ndarray, constants: np.ndarray
            ) -> np.ndarray:

                u = np.empty((_num_controls, len(independent)))
                for idx, (ti, xi, lami) in enumerate(zip(independent, states.T, costates.T)):
                    u[:, idx] = _compute_control(ti, xi, lami, parameters, constants)

                return u

            prob.control_handler.compute_control_vectorized = jit_compile(
                    _compute_control_vectorized,
                    (NumbaArray, NumbaMatrix, NumbaMatrix, NumbaArray, NumbaArray)
            )

            prob.control_handler = cast(VectorizedAlgebraicControlHandler, prob.control_handler)

        elif isinstance(prob.control_handler, DifferentialControlHandler):
            _compute_control_dynamics = prob.control_handler.compute_control_dynamics

            def _compute_control_dynamics_vectorized(
                    independent: np.ndarray, states: np.ndarray, costates: np.ndarray, controls: np.ndarray,
                    parameters: np.ndarray, constants: np.ndarray
            ) -> np.ndarray:

                u_dot = np.empty_like(controls)
                for idx, (ti, xi, lami, ui) in enumerate(zip(independent, states.T, costates.T, controls.T)):
                    u_dot[:, idx] = _compute_control_dynamics(ti, xi, lami, ui, parameters, constants)

                return u_dot

            prob.control_handler.compute_control_dynamics_vectorized = jit_compile(
                    _compute_control_dynamics_vectorized,
                    (NumbaArray, NumbaMatrix, NumbaMatrix, NumbaMatrix, NumbaArray, NumbaArray)
            )

            prob.control_handler = cast(VectorizedDifferentialControlHandler, prob.control_handler)

    return prob
