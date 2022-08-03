from typing import Union, Optional, Callable, TypeVar

from numpy.typing import ArrayLike

from giuseppe.io import Solution
from giuseppe.problems import CompBVP, CompOCP, CompDualOCP, AdiffBVP, AdiffOCP, AdiffDualOCP
from giuseppe.problems.dual.utils import sift_ocp_and_dual
from .simple import propagate_guess
from ..constant import initialize_guess_for_auto
from ..projection import match_states_to_bc, match_costates_to_bc

_IVP_SOL = TypeVar('_IVP_SOL')
CONTROL_FUNC = Callable[[float, ArrayLike, ArrayLike, ArrayLike], ArrayLike]
SUPPORTED_PROBLEMS = Union[CompBVP, CompOCP, CompDualOCP, AdiffBVP, AdiffOCP, AdiffDualOCP]


# TODO Allow for reverse integration
def auto_propagate_guess(
        comp_prob: SUPPORTED_PROBLEMS, default: Union[float, Solution] = 0.1,
        t_span: Union[float, ArrayLike] = 0.1,
        initial_states: Optional[ArrayLike] = None, initial_costates: Optional[ArrayLike] = None,
        control: Optional[Union[float, ArrayLike, CONTROL_FUNC]] = None, p: Optional[Union[float, ArrayLike]] = None,
        k: Optional[Union[float, ArrayLike]] = None, use_project_dual: bool = True, use_match_constants: bool = True,
        reverse: bool = False, abs_tol: float = 1e-3, rel_tol: float = 1e-3) -> Solution:

    """
    Propagate a guess with a constant control value or control function.

    After propagation, projection may be used to estimate the dual variable (costates and adjoints) and match the
    constants to the guess.

    Unspecified values will be set to default.

    Parameters
    ----------
    comp_prob : CompBVP, CompOCP, or CompDualOCP
        the problem that the guess is for
    default : float, BVPSol, OCPSol or DualOCPSol, default=0.1
        value used if no value is given
    t_span : float or ArrayLike, default=0.1
        values for the independent variable, t
        if float, t = np.array([0., t_span])
    initial_states : ArrayLike, optional
        state values to start propagation
    initial_costates : ArrayLike, optional
        costate values to start propagation, might change with use_project_dual=True
    control : float, Arraylike, or Callable[[float, ArrayLike, ArrayLike, Arraylike], ArrayLike]
        control used in propagation
        if float, all control values are set to that value
        if ArrayLike, control is set to constant specified array
        if Callable, callable function is called to compute control at each time
    p : ArrayLike, optional
        parameter values
    k : ArrayLike, optional
        constant values
        updated at end if use_match_constants=True
    use_match_constants : bool, default=True
        if True, match_constants will be called to update the constants to most closely match the formed guess
    use_project_dual : bool, default=True
        if True and prob is an instance CompDualOCP, project_dual will be called to estimate dual values from guess
    abs_tol : float, default=1e-3
       absolute tolerance
    rel_tol : float, default=1e-3
       relative tolerance

    Returns
    -------
    guess

    """
    prob, dual = sift_ocp_and_dual(comp_prob)

    if hasattr(default, '__float__'):
        guess = initialize_guess_for_auto(comp_prob, t_span=t_span, constants=k, default=default)
    elif isinstance(default, Solution):
        guess = default
    else:
        raise TypeError(f'default should be float or a solution type, not {type(default)}')

    if not reverse:
        location = 'initial'
    else:
        location = 'terminal'

    if initial_states is None:
        initial_states = match_states_to_bc(prob, guess, location=location, rel_tol=rel_tol, abs_tol=abs_tol)

    if (isinstance(comp_prob, CompDualOCP) or isinstance(comp_prob, AdiffDualOCP)) and initial_costates is None:
        initial_costates = match_costates_to_bc(dual, guess, states=initial_states, rel_tol=rel_tol, abs_tol=abs_tol)

    guess = propagate_guess(
            comp_prob, default=guess, t_span=t_span, initial_states=initial_states, initial_costates=initial_costates,
            control=control, p=p, k=k, use_project_dual=use_project_dual, use_match_constants=use_match_constants,
            reverse=reverse, abs_tol=abs_tol, rel_tol=rel_tol
    )

    return guess
