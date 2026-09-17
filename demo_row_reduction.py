import numpy as np
from row_reduction import row_echelon, reduced_row_echelon


def display_test(name, A):
    print("\n" + "=" * 60)
    print(f"  TEST MATRIX {name}")
    print("=" * 60)

    print("\nOriginal:")
    print(A)

    print("\nRow-echelon form:")
    print(row_echelon(A))

    print("\nReduced row-echelon form:")
    print(reduced_row_echelon(A))

    print()


# A: Basic invertible matrix
A = np.array([[1, 2], [3, 4]])

# B: Rectangular Matrix
B = np.array([[1, 2, 3, 4], [2, 4, 7, 10], [3, 6, 10, 14]])

# Skipped pivot column
C = np.array([[0, 2, 4, 6], [0, -10, 3, 1], [0, 5, 7, 8]])

# D: Zero matrix
D = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

# E: Tall matrix
E = np.array([[1, 2], [2, 4], [3, 5], [4, 7]])

# F: Rank-deficient matrix with dependent rows
F = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])

# G: Tiny initial pivot — tests partial pivoting
G = np.array([[1e-12, 1, 2], [5, 2, 3], [-10, 4, 1]])


display_test("A — Basic Invertible Matrix", A)
display_test("B — Rectangular Matrix", B)
display_test("C — Skipped Pivot Column", C)
display_test("D — Zero Matrix", D)
display_test("E — Tall Matrix", E)
display_test("F — Rank-Deficient Matrix", F)
display_test("G — Tiny Pivot / Partial Pivoting", G)
