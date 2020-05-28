"""Testing operations of quantum states."""
import numpy as np

from toqito.state_ops import pure_to_mixed
from toqito.state_ops import schmidt_decomposition

from toqito.states import bell
from toqito.states import max_entangled


def test_pure_to_mixed_state_vector():
    """Convert pure state to mixed state vector."""
    expected_res = np.array(
        [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]
    )

    phi = bell(0)
    res = pure_to_mixed(phi)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_pure_to_mixed_density_matrix():
    """Convert pure state to mixed state density matrix."""
    expected_res = np.array(
        [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]
    )

    phi = bell(0) * bell(0).conj().T
    res = pure_to_mixed(phi)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_invalid_pure_to_mixed_input():
    """Invalid arguments for pure_to_mixed."""
    with np.testing.assert_raises(ValueError):
        non_valid_input = np.array([[1 / 2, 0, 0, 1 / 2], [1 / 2, 0, 0, 1 / 2]])
        pure_to_mixed(non_valid_input)


def test_schmidt_decomp_max_ent():
    """Schmidt decomposition of the 3-D maximally entangled state."""
    singular_vals, u_mat, vt_mat = schmidt_decomposition(max_entangled(3))

    expected_u_mat = np.identity(3)
    expected_vt_mat = np.identity(3)
    expected_singular_vals = 1 / np.sqrt(3) * np.array([[1], [1], [1]])

    bool_mat = np.isclose(expected_u_mat, u_mat)
    np.testing.assert_equal(np.all(bool_mat), True)

    bool_mat = np.isclose(expected_vt_mat, vt_mat)
    np.testing.assert_equal(np.all(bool_mat), True)

    bool_mat = np.isclose(expected_singular_vals, singular_vals)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_schmidt_decomp_dim_list():
    """Schmidt decomposition with list specifying dimension."""
    singular_vals, u_mat, vt_mat = schmidt_decomposition(max_entangled(3), dim=[3, 3])

    expected_u_mat = np.identity(3)
    expected_vt_mat = np.identity(3)
    expected_singular_vals = 1 / np.sqrt(3) * np.array([[1], [1], [1]])

    bool_mat = np.isclose(expected_u_mat, u_mat)
    np.testing.assert_equal(np.all(bool_mat), True)

    bool_mat = np.isclose(expected_vt_mat, vt_mat)
    np.testing.assert_equal(np.all(bool_mat), True)

    bool_mat = np.isclose(expected_singular_vals, singular_vals)
    np.testing.assert_equal(np.all(bool_mat), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
