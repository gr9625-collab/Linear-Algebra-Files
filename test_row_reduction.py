import numpy as np
from row_reduction import reduced_row_echelon

# ============================================================
# TEST A — Basic Invertible Matrix
# ============================================================

A = np.array([[1, 2], [3, 4]])

expected_A = np.array([[1, 0], [0, 1]])

assert np.allclose(
    reduced_row_echelon(A), expected_A
), "Test A failed: basic invertible matrix"


# ============================================================
# TEST B — Rectangular Matrix
# ============================================================

B = np.array([[1, 2, 3, 4], [2, 4, 7, 10], [3, 6, 10, 14]])

expected_B = np.array([[1, 2, 0, -2], [0, 0, 1, 2], [0, 0, 0, 0]])

assert np.allclose(
    reduced_row_echelon(B), expected_B
), "Test B failed: rectangular matrix"


# ============================================================
# TEST C — Skipped Pivot Column
# ============================================================

C = np.array([[0, 2, 4, 6], [0, -10, 3, 1], [0, 5, 7, 8]])

expected_C = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

assert np.allclose(
    reduced_row_echelon(C), expected_C
), "Test C failed: skipped pivot column"


# ============================================================
# TEST D — Zero Matrix
# ============================================================

D = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

expected_D = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

assert np.allclose(reduced_row_echelon(D), expected_D), "Test D failed: zero matrix"


# ============================================================
# TEST E — Tall Matrix
# ============================================================

E = np.array([[1, 2], [2, 4], [3, 5], [4, 7]])

expected_E = np.array([[1, 0], [0, 1], [0, 0], [0, 0]])

assert np.allclose(reduced_row_echelon(E), expected_E), "Test E failed: tall matrix"


# ============================================================
# TEST F — Rank-Deficient Matrix
# ============================================================

F = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])

expected_F = np.array([[1, 2, 3], [0, 0, 0], [0, 0, 0]])

assert np.allclose(
    reduced_row_echelon(F), expected_F
), "Test F failed: rank-deficient matrix"


# ============================================================
# TEST G — Tiny Pivot / Partial Pivoting
# ============================================================

G = np.array([[1e-12, 1, 2], [5, 2, 3], [-10, 4, 1]])

expected_G = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

assert np.allclose(
    reduced_row_echelon(G), expected_G
), "Test G failed: tiny pivot / partial pivoting"


# ============================================================
# ALL TESTS PASSED
# ============================================================

print("=" * 60)
print("ALL ROW REDUCTION TESTS PASSED")
print("=" * 60)
