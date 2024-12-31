"""Two-player extended nonlocal game."""

from collections import defaultdict

import cvxpy
import numpy as np

from toqito.helper import npa_constraints, update_odometer
from toqito.matrix_ops import tensor
from toqito.rand import random_unitary


class ExtendedNonlocalGame:
    r"""Create two-player extended nonlocal game object.

    *Extended nonlocal games* are a superset of nonlocal games in which the
    players share a tripartite state with the referee. In such games, the
    winning conditions for Alice and Bob may depend on outcomes of measurements
    made by the referee, on its part of the shared quantum state, in addition
    to Alice and Bob's answers to the questions sent by the referee.

    Extended nonlocal games were initially defined in :cite:`Johnston_2016_Extended` and more
    information on these games can be found in :cite:`Russo_2017_Extended`.

    An example demonstration is available as a tutorial in the
    documentation. Go to :ref:`ref-label-bb84_extended_nl_example`.

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    """

    def __init__(self, prob_mat: np.ndarray, pred_mat: np.ndarray, reps: int = 1) -> None:
        """Construct extended nonlocal game object.

        :param prob_mat: A matrix whose (x, y)-entry gives the probability
                        that the referee will give Alice the value `x` and Bob
                        the value `y`.
        :param pred_mat: A matrix whose (...)-entry gives ...
        :param reps: Number of parallel repetitions to perform.
        """
        if reps == 1:
            self.prob_mat = prob_mat
            self.pred_mat = pred_mat
            self.reps = reps

        else:
            (
                dim_x,
                dim_y,
                A,
                B,
                X,
                Y,
            ) = pred_mat.shape
            self.prob_mat = tensor(prob_mat, reps)

            pred_mat2 = np.zeros(
                (
                    dim_x**reps,
                    dim_y**reps,
                    A**reps,
                    B**reps,
                    X**reps,
                    Y**reps,
                )
            )
            i_ind = np.zeros(reps, dtype=int)
            j_ind = np.zeros(reps, dtype=int)
            for i in range(X**reps):
                for j in range(Y**reps):
                    to_tensor = np.empty([reps, dim_x, dim_y, A, B])
                    for k in range(reps - 1, -1, -1):
                        to_tensor[k] = pred_mat[:, :, :, :, i_ind[k], j_ind[k]]
                    pred_mat2[:, :, :, :, i, j] = tensor(to_tensor)
                    j_ind = update_odometer(j_ind, Y * np.ones(reps))
                i_ind = update_odometer(i_ind, X * np.ones(reps))
            self.pred_mat = pred_mat2
            self.reps = reps

    def unentangled_value(self) -> float:
        r"""Calculate the unentangled value of an extended nonlocal game.

        The *unentangled value* of an extended nonlocal game is the supremum
        value for Alice and Bob's winning probability in the game over all
        unentangled strategies. Due to convexity and compactness, it is possible
        to calculate the unentangled extended nonlocal game by:

        .. math::
            \omega(G) = \max_{f, g}
            \lVert
            \sum_{(x,y) \in \Sigma_A \times \Sigma_B} \pi(x,y)
            V(f(x), g(y)|x, y)
            \rVert

        where the maximum is over all functions :math:`f : \Sigma_A \rightarrow
        \Gamma_A` and :math:`g : \Sigma_B \rightarrow \Gamma_B`.

        :return: The unentangled value of the extended nonlocal game.
        """
        dim_x, dim_y, A, B, X, Y = self.pred_mat.shape

        max_unent_val = float("-inf")
        for a_out in range(A):
            for b_out in range(B):
                p_win = np.zeros([dim_x, dim_y], dtype=complex)
                for x_in in range(X):
                    for y_in in range(Y):
                        p_win += self.prob_mat[x_in, y_in] * self.pred_mat[:, :, a_out, b_out, x_in, y_in]

                rho = cvxpy.Variable((dim_x, dim_y), hermitian=True)

                objective = cvxpy.Maximize(cvxpy.real(cvxpy.trace(p_win.conj().T @ rho)))

                constraints = [cvxpy.trace(rho) == 1, rho >> 0]
                problem = cvxpy.Problem(objective, constraints)
                unent_val = problem.solve()
                max_unent_val = max(max_unent_val, unent_val)
        return max_unent_val

    def nonsignaling_value(self) -> float:
        r"""Calculate the non-signaling value of an extended nonlocal game.

        The *non-signaling value* of an extended nonlocal game is the supremum
        value of the winning probability of the game taken over all
        non-signaling strategies for Alice and Bob.

        A *non-signaling strategy* for an extended nonlocal game consists of a
        function

        .. math::
            K : \Gamma_A \times \Gamma_B \times \Sigma_A \times \Sigma_B
            \rightarrow \text{Pos}(\mathcal{R})

        such that

        .. math::
            \sum_{a \in \Gamma_A} K(a,b|x,y) = \rho_b^y
            \quad \text{and} \quad
            \sum_{b \in \Gamma_B} K(a,b|x,y) = \sigma_a^x,

        for all :math:`x \in \Sigma_A` and :math:`y \in \Sigma_B` where
        :math:`\{\rho_b^y : y \in \Sigma_A, \ b \in \Gamma_B\}` and
        :math:`\{\sigma_a^x : x \in \Sigma_A, \ a \in \Gamma_B\}` are
        collections of operators satisfying

        .. math::
            \sum_{a \in \Gamma_A} \rho_b^y =
            \tau =
            \sum_{b \in \Gamma_B} \sigma_a^x,

        for every choice of :math:`x \in \Sigma_A` and :math:`y \in \Sigma_B`
        where :math:`\tau \in \text{D}(\mathcal{R})` is a density operator.

        :return: The non-signaling value of the extended nonlocal game.
        """
        dim_x, dim_y, A, B, X, Y = self.pred_mat.shape
        constraints = []

        # The cvxpy package does not support optimizing over more than
        # 2-dimensional objects. To overcome this, we use a dictionary to index
        # between the questions and answers, while the cvxpy variables held at
        # this positions are `dim_x`-by-`dim_y` cvxpy Variable objects.

        # Define K(a,b|x,y) variable.
        K = defaultdict(cvxpy.Variable)  # assemblage
        for a in range(A):
            for b in range(B):
                for x in range(X):
                    for y in range(Y):
                        K[a, b, x, y] = cvxpy.Variable((dim_x, dim_y), hermitian=True)
                        constraints.append(K[a, b, x, y] >> 0)

        # Define \sigma_a^x variable.
        sigma = defaultdict(cvxpy.Variable)
        for a in range(A):
            for x in range(X):
                sigma[a, x] = cvxpy.Variable((dim_x, dim_y), hermitian=True)

        # Define \rho_b^y variable.
        rho = defaultdict(cvxpy.Variable)
        for b in range(B):
            for y in range(Y):
                rho[b, y] = cvxpy.Variable((dim_x, dim_y), hermitian=True)

        # Define \tau density operator.
        tau = cvxpy.Variable((dim_x, dim_y), hermitian=True)
        constraints.append(cvxpy.trace(tau) == 1)
        constraints.append(tau >> 0)

        p_win = 0
        for a in range(A):
            for b in range(B):
                for x in range(X):
                    for y in range(Y):
                        p_win += self.prob_mat[x, y] * cvxpy.trace(
                            self.pred_mat[:, :, a, b, x, y].conj().T @ K[a, b, x, y]
                        )

        objective = cvxpy.Maximize(cvxpy.real(p_win))

        # The following constraints enforce the so-called non-signaling
        # constraints.

        # Enforce that:
        # \sum_{b \in \Gamma_B} K(a,b|x,y) = \sigma_a^x
        for x in range(X):
            for y in range(Y):
                for a in range(A):
                    b_sum = 0
                    for b in range(B):
                        b_sum += K[a, b, x, y]
                    constraints.append(b_sum == sigma[a, x])

        # Enforce non-signaling constraints on Alice marginal:
        # \sum_{a \in \Gamma_A} K(a,b|x,y) = \rho_b^y
        for x in range(X):
            for y in range(Y):
                for b in range(B):
                    a_sum = 0
                    for a in range(A):
                        a_sum += K[a, b, x, y]
                    constraints.append(a_sum == rho[b, y])

        # Enforce non-signaling constraints on Bob marginal:
        # \sum_{a \in \Gamma_A} \sigma_a^x = \tau
        for x in range(X):
            sig_a_sum = 0
            for a in range(A):
                sig_a_sum += sigma[a, x]
            constraints.append(sig_a_sum == tau)

        # Enforce that:
        # \sum_{b \in \Gamma_B} \rho_b^y = \tau
        for y in range(Y):
            rho_b_sum = 0
            for b in range(B):
                rho_b_sum += rho[b, y]
            constraints.append(rho_b_sum == tau)

        problem = cvxpy.Problem(objective, constraints)
        ns_val = problem.solve()

        return ns_val

    def quantum_value_lower_bound(self, iters: int = 5, tol: float = 10e-6) -> float:
        r"""Calculate lower bound on the quantum value of an extended nonlocal game.

        Test

        :return: The quantum value of the extended nonlocal game.
        """
        # Get number of inputs and outputs for Bob's measurements.
        _, _, _, B, _, Y = self.pred_mat.shape

        best_lower_bound = float("-inf")
        for _ in range(1):
            # Generate a set of random POVMs for Bob. These measurements serve as a
            # rough starting point for the alternating projection algorithm.
            bob_povms = defaultdict(int)
            for y in range(Y):
                u = random_unitary(B)

                for b in range(B):
                    ut = u[:, b].conj().T.reshape(-1, 1)
                    bob_povms[y, b] = u[:, b] * ut

            # Run the alternating projection algorithm between the two SDPs.
            it_diff = 1
            prev_win = -1
            best = float("-inf")
            while it_diff > tol:
                # Optimize over Alice's measurement operators while fixing Bob's.
                # If this is the first iteration, then the previously randomly
                # generated operators in the outer loop are Bob's. Otherwise, Bob's
                # operators come from running the next SDP.
                rho, lower_bound_a = self.__optimize_alice(bob_povms)
                bob_povms, lower_bound_b = self.__optimize_bob(rho)

                lower_bound = max(lower_bound_a, lower_bound_b)

                it_diff = lower_bound - prev_win
                prev_win = lower_bound
                # As the SDPs keep alternating, check if the winning probability
                # becomes any higher. If so, replace with new best.
                best = max(best, lower_bound)

            best_lower_bound = max(best, best_lower_bound)

        return best_lower_bound

    def __optimize_alice(self, povms) -> tuple[dict, float]:
        """Fix Bob's measurements and optimize over Alice's measurements."""
        # Get number of inputs and outputs.
        (
            dim,
            _,
            A,
            B,
            X,
            Y,
        ) = self.pred_mat.shape

        # The cvxpy package does not support optimizing over 4-dimensional objects.
        # To overcome this, we use a dictionary to index between the questions and
        # answers, while the cvxpy variables held at this positions are
        # `dim`-by-`dim` cvxpy variables.
        rho = defaultdict(cvxpy.Variable)
        for x, a in np.ndindex((X, A)):
            rho[x, a] = cvxpy.Variable((dim * B, dim * B), hermitian=True)

        tau = cvxpy.Variable((dim * B, dim * B), hermitian=True)
        win = 0
        for x, y, a, b in np.ndindex((X, Y, A, B)):
            win += self.prob_mat[x, y] * cvxpy.trace(
                (
                    np.kron(
                        self.pred_mat[:, :, a, b, x, y],
                        povms[y, b] if isinstance(povms[y, b], np.ndarray) else povms[y, b].value,
                    )
                )
                .conj()
                .T
                @ rho[x, a]
            )

        objective = cvxpy.Maximize(cvxpy.real(win))
        constraints = []

        # Sum over "a" for all "x" for Alice's measurements.
        for x in range(X):
            rho_sum_a = 0
            for a in range(A):
                rho_sum_a += rho[x, a]
                constraints.append(rho[x, a] >> 0)

            constraints.append(rho_sum_a == tau)

        constraints.append(cvxpy.trace(tau) == 1)
        constraints.append(tau >> 0)

        problem = cvxpy.Problem(objective, constraints)

        lower_bound = problem.solve()
        return rho, lower_bound

    def __optimize_bob(self, rho) -> tuple[dict, float]:
        """Fix Alice's measurements and optimize over Bob's measurements."""
        # Get number of inputs and outputs.
        (
            dim,
            _,
            A,
            B,
            X,
            Y,
        ) = self.pred_mat.shape

        # The cvxpy package does not support optimizing over 4-dimensional objects.
        # To overcome this, we use a dictionary to index between the questions and
        # answers, while the cvxpy variables held at this positions are
        # `dim`-by-`dim` cvxpy variables.
        bob_povms = defaultdict(cvxpy.Variable)
        for y, b in np.ndindex((Y, B)):
            bob_povms[y, b] = cvxpy.Variable((dim, dim), hermitian=True)
        win = 0
        for x, y, a, b in np.ndindex((X, Y, A, B)):
            win += self.prob_mat[x, y] * cvxpy.trace(
                (
                    cvxpy.kron(
                        self.pred_mat[:, :, a, b, x, y],
                        bob_povms[y, b],
                    )
                )
                @ rho[x, a].value
            )
        objective = cvxpy.Maximize(cvxpy.real(win))

        constraints = []

        # Sum over "b" for all "y" for Bob's measurements.
        for y in range(Y):
            bob_sum_b = 0
            for b in range(B):
                bob_sum_b += bob_povms[y, b]
                constraints.append(bob_povms[y, b] >> 0)
            constraints.append(bob_sum_b == np.identity(B))

        problem = cvxpy.Problem(objective, constraints)

        lower_bound = problem.solve()
        return bob_povms, lower_bound

    def commuting_measurement_value_upper_bound(self, k: int | str = 1) -> float:
        """Compute an upper bound on the commuting measurement value of an extended nonlocal game.

        This function calculates an upper bound on the commuting measurement value by
        using k-levels of the NPA hierarchy :cite:`Navascues_2008_AConvergent`. The NPA hierarchy is a uniform family
        of semidefinite programs that converges to the commuting measurement value of
        any extended nonlocal game.

        You can determine the level of the hierarchy by a positive integer or a string
        of a form like '1+ab+aab', which indicates that an intermediate level of the hierarchy
        should be used, where this example uses all products of one measurement, all products of
        one Alice and one Bob measurement, and all products of two Alice and one Bob measurements.

        References
        ==========
        .. bibliography::
            :filter: docname in docnames

        :param k: The level of the NPA hierarchy to use (default=1).
        :return: The upper bound on the commuting strategy value of an extended nonlocal game.

        """
        m, _, A, B, X, Y = self.pred_mat.shape

        # Our pseudo commuting measurement assemblage.
        K = defaultdict(cvxpy.Variable)
        for x, y in np.ndindex((X, Y)):
            K[x, y] = cvxpy.Variable(
                (m * A, m * B),
                name=f"K(A, B | {x}, {y})",
                hermitian=True,
            )

        # omega = max_{K} \sum_{ABXY} \pi(x,y) <V, K>
        omega = cvxpy.Constant(0)
        for a, b, x, y in np.ndindex((A, B, X, Y)):
            omega += self.prob_mat[x, y] * cvxpy.trace(
                self.pred_mat[:, :, a, b, x, y].conj().T
                @ K[x, y][
                    a * m : (a + 1) * m,
                    b * m : (b + 1) * m,
                ]
            )

        return cvxpy.Problem(cvxpy.Maximize(cvxpy.real(omega)), npa_constraints(K, k, m)).solve()
