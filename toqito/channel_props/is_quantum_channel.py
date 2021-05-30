"""Is quantum channel."""
from typing import List, Union

import numpy as np

from toqito.channel_ops import kraus_to_choi
from toqito.channel_props import is_completely_positive
from toqito.channel_props import is_trace_preserving


def is_quantum_channel(
    phi: Union[np.ndarray, List[List[np.ndarray]]],
    rtol: float = 1e-05,
    atol: float = 1e-08,
) -> bool:
    r"""
    Determine whether the given input is a quantum channel [WatQC18]_.
    
    A map :math:`\Phi \in \text{T} \left(\mathcal{X}, \mathcal{Y} \right)` is a *quantum
    channel* for some choice of complex Euclidean spaces :math:`\mathcal{X}`
    and :math:`\mathcal{Y}`, if it holds that:

    1. :math:`\Phi` is completely positive.
    2. :math:`\Phi` is trace preserving.

    Examples
    ==========
    We can specify the input as a list of Kraus operators. Consider the map :math:`\Phi` defined as

    .. math::
        \Phi(X) = X - U X U^*

    where

    .. math::
        U = \frac{1}{\sqrt{2}}
        \begin{pmatrix}
            1 & 1 \\
            -1 & 1
        \end{pmatrix}.

    This map is not a quantum channel, as we can verify as follows.

    >>> from toqito.channel_props import is_quantum_channel
    >>> import numpy as np
    >>> unitary_mat = np.array([[1, 1], [-1, 1]]) / np.sqrt(2)
    >>> kraus_ops = [[np.identity(2), np.identity(2)], [unitary_mat, -unitary_mat]]
    >>> is_quantum_channel(kraus_ops)
    False

    We can also specify the input as a Choi matrix. For instance, consider the Choi matrix
    corresponding to the :math:`2`-dimensional completely depolarizing channel

    .. math::
        \Omega =
        \frac{1}{2}
        \begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & 1 & 0 \\
            0 & 0 & 0 & 1
        \end{pmatrix}.

    We may verify that this channel is a quantum channel

    >>> from toqito.channels import depolarizing
    >>> from toqito.channel_props import is_quantum_channel
    >>> is_completely_positive(depolarizing(2))
    True

    References
    ==========
    .. [WatQC18] Watrous, John.
        "The Theory of Quantum Information."
        Section: "2.2.1  Definitions and basic notions concerning channels".
        Cambridge University Press, 2018.
    :param phi: The channel provided as either a Choi matrix or a list of Kraus operators.
    :param rtol: The relative tolerance parameter (default 1e-05).
    :param atol: The absolute tolerance parameter (default 1e-08).
    :return: True if the channel is a quantum channel, and False otherwise.
    """
    # If the variable `phi` is provided as a list, we assume this is a list
    # of Kraus operators.
    if isinstance(phi, list):
        phi = kraus_to_choi(phi)

    # A valid quantum channel is a superoperator that is both completely
    # positive and trace-preserving
    return is_completely_positive(phi, rtol, atol) and is_trace_preserving(phi, rtol, atol)
