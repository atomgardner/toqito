"""Determines whether or not a channel is Hermitian-preserving."""
from typing import List, Union
import numpy as np

from toqito.channels.operations.kraus_to_choi import kraus_to_choi


def is_herm_preserving(
    phi: Union[np.ndarray, List[List[np.ndarray]]], tol: float = 1e-05
) -> bool:
    r"""
    Determine whether the given channel is Hermitian-preserving [WatH18]_.

    A map :math:`\Phi \in \text{T} \left(\mathcal{X}, \mathcal{Y} \right)` is
    Hermitian-preserving if it holds that

    .. math::
        \Phi(H) \in \text{Herm}(\mathcal{Y})

    for every Hermitian operator :math:`H \in \text{Herm}(\mathcal{X}`.

    Examples
    ==========

    The map :math:`\Phi` defined as

    .. math::
        \Phi(X) = X - U X U^*

    is Hermitian-preserving, where

    .. math::
        U = \frac{1}{\sqrt{2}}
        \begin{pmatrix}
            1 & 1 \\
            -1 & 1
        \end{pmatrix}.

    >>> import numpy as np
    >>> from toqito.channels.properties.is_herm_preserving import is_herm_preserving
    >>> unitary_mat = np.array([[1, 1], [-1, 1]]) / np.sqrt(2)
    >>> kraus_ops = [[np.identity(2), np.identity(2)], [unitary_mat, -unitary_mat]]
    >>> print(is_herm_preserving(kraus_ops))
    True

    We may also verify whether the corresponding Choi matrix of a given map is
    Hermitian-preserving. The swap operator is the Choi matrix of the transpose
    map, which is Hermitian-preserving as can be seen as follows:

    >>> import numpy as np
    >>> from toqito.perms.swap_operator import swap_operator
    >>> from toqito.channels.properties.is_herm_preserving import is_herm_preserving
    >>> unitary_mat = np.array([[1, 1], [-1, 1]]) / np.sqrt(2)
    >>> choi_mat = swap_operator(3)
    >>> print(is_herm_preserving(choi_mat))
    True

    References
    ==========
    .. [WatH18] Watrous, John.
        "The theory of quantum information."
        Section: "Linear maps of square operators".
        Cambridge University Press, 2018.

    :param phi: The channel provided as either a Choi matrix or a list of
                Kraus operators.
    :param tol: The tolerance parameter to determine Hermiticity.
    :return: True if the channel is Hermitian-preserving, and False otherwise.
    """
    # If the variable `phi` is provided as a list, we assume this is a list
    # of Kraus operators.
    if isinstance(phi, list):
        phi = kraus_to_choi(phi)

    # Phi is Hermiticity-preserving iff its Choi matrix is Hermitian.
    if phi.shape[0] != phi.shape[1]:
        return False
    return np.max(np.max(np.abs(phi - phi.conj().T))) <= tol
