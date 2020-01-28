"""Produces an isotropic state."""
import numpy as np
from toqito.states.max_entangled import max_entangled
from scipy.sparse import identity


def isotropic_state(dim: int, alpha: float) -> np.ndarray:
    """
    Produces a Isotropic state.
    :param dim: The local dimension.
    :param alpha: The parameter of the isotropic state.
    :return: Isotropic state.

    Returns the isotropic state with parameter ALPHA acting on
    (DIM*DIM)-dimensional space. More specifically, the state is the density
    operator defined by (1-ALPAH)*I/DIM^2 + ALPHAE, where I is the identity
    operator and E is the projection onto the standard maximally-entangled
    pure state on two copies of DIM-dimensional space.

    References:
    [1] N. Gisin. Hidden quantum nonlocality revealed by local filters.
        (http://dx.doi.org/10.1016/S0375-9601(96)80001-6). 1996.

    """
    # Compute the isotropic state.
    psi = max_entangled(dim, True, False)
    return (1 - alpha) * identity(dim**2)/dim**2 + alpha*psi*psi.conj().T/dim
