"""Generates the NPA constraints."""

from collections import namedtuple
from itertools import product

import cvxpy
from numpy import ndindex

Symbol = namedtuple("Symbol", ["player", "question", "answer"], defaults=["", None, None])


# This function simplifies the input word by applying
# the commutation and projection rules.
def _reduce(word: tuple[Symbol]) -> tuple[Symbol]:
    # commute: bring Alice in front.
    s, t = (), ()
    for u in word:
        if u.player == "Alice":
            s += (u,)
        if u.player == "Bob":
            t += (u,)

    return _reduce_proj(s + t)


# use the projection rules to simplify an input word.
def _reduce_proj(word: tuple[Symbol]) -> tuple[Symbol]:
    for i in range(len(word) - 1):
        s, t = word[i], word[i + 1]

        # orthogonal.
        if s.player == t.player and s.question == t.question and s.answer != t.answer:
            return ()

        # idempotent.
        if s == t:
            return _reduce_proj(word[:i] + word[i + 1 :])

    return word


def _parse(k: str) -> tuple[int, set[tuple[int, int]]]:
    k = k.split("+")
    base_k = int(k[0])

    conf = set()
    for val in k[1:]:
        # otherwise we already take this configuration
        # in base_k - level of hierarchy.
        if len(val) > base_k:
            cnt_a, cnt_b = 0, 0
            for bit in val:
                if bit == "a":
                    cnt_a += 1
                if bit == "b":
                    cnt_b += 1

            conf.add((cnt_a, cnt_b))

    return base_k, conf


# This function generates all non-equivalent words in X×A ⊔ Y×B with length at most k.
def _gen_words(k: int | str, A: int, X: int, B: int, Y: int) -> list[tuple[Symbol]]:
    words = [(Symbol(""),)]

    # Measurements sum to identity, so we can remove one outcome.
    # XXX(tom): how is this enforced?
    As = [Symbol("Alice", x, a) for x in range(X) for a in range(A - 1)]
    Bs = [Symbol("Bob", y, b) for y in range(Y) for b in range(B - 1)]

    conf = []
    if isinstance(k, str):
        k, conf = _parse(k)

    # Generate strings of the form:
    #     A^{x_1}_{a_1}...A^{x_j}_{a_j} B^{y_1}_{b_1}...B^{y_(i-j)}_{b_(i-j)}
    # In words this says: generate strings of length i where Alice does j-many measurements.
    words += [
        s + t
        for i in range(1, k + 1)
        for j in range(i + 1)
        for s in product(As, repeat=j)
        if len(_reduce(s)) == j  # XXX: are we sure about this?
        for t in product(Bs, repeat=i - j)
        if len(_reduce(t)) == i - j
    ]

    # Generate the intermediate levels of hierarchy (eg, 2+aab+aabb)
    words += [
        s + t
        for a, b in conf
        for s in product(As, repeat=a)
        if len(_reduce(s)) == a
        for t in product(Bs, repeat=b)
        if len(_reduce(t)) == b
    ]

    return words


def _is_zero(word: tuple[Symbol]) -> bool:
    return len(word) == 0


def _is_meas(word: tuple[Symbol]) -> bool:
    if len(word) == 2:
        s_a, s_b = word
        return s_a.player == "Alice" and s_b.player == "Bob"

    return False


def _is_meas_on_one_player(word: tuple[Symbol]) -> bool:
    return len(word) == 1 and word[0].player in {"Alice", "Bob"}


def _get_nonlocal_game_params(K: dict[tuple[int, int], cvxpy.Variable], m: int = 1) -> tuple[int, int, int, int]:
    X, Y = max(K.keys())
    X = X + 1
    Y = Y + 1

    operator = next(iter(K.values()))
    A = int(operator.shape[0] / m)
    B = int(operator.shape[1] / m)

    return A, X, B, Y


def npa_constraints(
    K: dict[tuple[int, int], cvxpy.Variable], k: int | str = 1, m: int = 1
) -> list[cvxpy.constraints.constraint.Constraint]:
    r"""Generate the constraints specified by the NPA hierarchy up to a finite level :cite:`Navascues_2008_AConvergent`.

    You can determine the level of the hierarchy by a positive integer or a string
    of a form like "1+ab+aab", which indicates that an intermediate level of the hierarchy
    should be used, where this example uses all products of 1 measurement, all products of
    one Alice and one Bob measurement, and all products of two Alice and one Bob measurement.

    The commuting measurement assemblage operator must be given as a dictionary. The keys are
    tuples of Alice and Bob questions :math:`x, y` and the values are cvxpy Variables which
    are matrices with entries:

    .. math::
        K_{xy}\Big(i + a \cdot dim_R, j + b \cdot dim_R \Big) =
        \langle i| \text{Tr}_{\mathcal{H}} \Big( \big(
            I_R \otimes A_a^x B_b^y \big) \sigma \Big) |j \rangle

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :param K: The commuting measurement assemblage operator.
    :param k: The level of the NPA hierarchy to use (default=1).
    :param m: The dimension of the referee's quantum system (default=1).
    :return: A list of cvxpy constraints.

    """
    A, X, B, Y = _get_nonlocal_game_params(K, m)
    words = _gen_words(k, A, X, B, Y)
    dim = len(words)

    # Certificate matrix.
    cert = cvxpy.Variable((m * dim, m * dim), hermitian=True, name="R")

    # The pseudo-commuting assemblage is built from these submatrices
    def M(i, j):
        return cert[i::dim, j::dim]

    # Normalization.
    # M_1,1 (ε, ε) + ... +  M_m,m (ε, ε) = 1
    constraints = [sum(M(i, i) for i in range(m)) == 1, cert >> 0]

    seen = {}
    for i in range(dim):
        for j in range(i, dim):
            word = _reduce(tuple(reversed(words[i])) + words[j])

            # Same question, different answer.
            #   /\ M((...)(x, a)(x, a')...) == 0
            #   /\ M((...)(y, b)(y, b')...) == 0
            # Note: i == 0 corresponds to (ε, ε).
            if i != 0 and _is_zero(word):
                constraints.append(M(i, j) == 0)

            # Marginals.
            #   /\ M(ε, s) = Σ_B M((0,b), s)
            #   /\ M(s, ε) = Σ_B M(s, (0,b))
            #   /\ M(ε, s) = Σ_A M((0,a), s)
            #   /\ M(s, ε) = Σ_A M(s, (0,a))
            # This constraint is extended to the other questions below.
            elif _is_meas_on_one_player(word):
                s = word[0]
                combo = (
                    sum(
                        K[s.question, 0][
                            m * s.answer : m * (s.answer + 1),
                            m * b : m * (b + 1),
                        ]
                        for b in range(B)
                    )
                    if s.player == "Alice"
                    else sum(
                        K[0, s.question][
                            m * a : m * (a + 1),
                            m * s.answer : m * (s.answer + 1),
                        ]
                        for a in range(A)
                    )
                )

                constraints.append(M(i, j) == combo)

            # K(a,b|x,y) = M((x,a), (y,b))
            elif _is_meas(word):
                s, t = word
                constraints.append(
                    K[s.question, t.question][
                        s.answer * m : (s.answer + 1) * m,
                        t.answer * m : (t.answer + 1) * m,
                    ]
                    == M(i, j)
                )

            elif word in seen:
                constraints.append(M(i, j) == M(*seen[word]))

            else:
                # The POVM elements for same player but different questions.
                seen[word] = (i, j)

    # Assemblage constraints.
    # K[ab|xy] is a density.
    for x, y in ndindex((X, Y)):
        sum_all_meas_and_trace = cvxpy.Constant(0)
        for a, b in ndindex((A, B)):
            # K[a,b|x,y] is PSD; it's an unnormalized state.
            constraints.append(
                K[x, y][
                    a * m : (a + 1) * m,
                    b * m : (b + 1) * m,
                ]
                >> 0
            )
            sum_all_meas_and_trace += sum(K[x, y][i + a * m, i + b * m] for i in range(m))

        # XXX: Tr(Σ_{A,B} K[ab|xy])) == 1 ???
        constraints.append(sum_all_meas_and_trace == 1)

    # Bob marginal consistency
    #   \A s,x>0: Σ_A M((0,a), s) = Σ_A M((x,a), s)
    for y, b in ndindex((Y, B)):
        first_marginal = sum(
            K[0, y][
                a * m : (a + 1) * m,
                b * m : (b + 1) * m,
            ]
            for a in range(A)
        )

        for x in range(1, X):
            marginal = sum(
                K[x, y][
                    a * m : (a + 1) * m,
                    b * m : (b + 1) * m,
                ]
                for a in range(A)
            )

            constraints.append(first_marginal == marginal)

    # Alice marginal consistency
    #   \A s,y>0: Σ_B M((0,b), s) = Σ_B M((y,b), s)
    for x, a in ndindex((X, A)):
        first_marginal = sum(
            K[x, 0][
                a * m : (a + 1) * m,
                b * m : (b + 1) * m,
            ]
            for b in range(B)
        )

        for y in range(1, Y):
            marginal = sum(
                K[x, y][
                    a * m : (a + 1) * m,
                    b * m : (b + 1) * m,
                ]
                for b in range(B)
            )

            constraints.append(first_marginal == marginal)

    return constraints
