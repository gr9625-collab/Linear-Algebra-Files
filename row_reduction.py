import numpy as np


# Row reduce an m by n matrix into row-echelon form
def row_echelon(B, eps=1e-10):
    # Copy the array so we don't accidentally change the original (and make it contain floats)
    A = np.array(B, dtype=float, copy=True)

    # The dimensions
    m = A.shape[0]
    n = A.shape[1]

    pivot_row = 0

    for j in range(n):
        # If the current column is all zeros from pivot row and below, move onto the next columns
        if np.all(np.abs(A[pivot_row:, j]) < eps):
            continue

        # Otherwise we can find a nonzero entry in the current column
        # Choose the entry at or below the pivot row with the largest magnitude
        nonzero_row = pivot_row + np.argmax(np.abs(A[pivot_row:, j]))

        # Then swap this row and the pivot row (if they are not already equal)
        if pivot_row != nonzero_row:
            temp_row = np.copy(A[pivot_row, :])
            A[pivot_row, :] = A[nonzero_row, :]
            A[nonzero_row, :] = temp_row

        # Now for all the rows below the pivot, row reduce them
        for i in range(pivot_row + 1, m):
            # Multiplication factor
            factor = A[i, j] / A[pivot_row, j]

            # Replace row i by (row i minus factor times the pivot row)
            A[i, :] -= factor * A[pivot_row, :]

        # Now move the pivot row up and stop if you have run out of rows
        pivot_row += 1
        if pivot_row == m:
            break

    # Set all small values to zero
    A[np.abs(A) < eps] = 0
    return A


# Row reduce an m by n matrix into reduced row-echelon form
def reduced_row_echelon(B, eps=1e-10):
    # Copy the array so we don't accidentally change the original (and make it contain floats)
    A = np.array(B, dtype=float, copy=True)
    A = row_echelon(A, eps)

    # The dimensions
    m = A.shape[0]
    n = A.shape[1]

    pivot = 0

    for i in range(m):
        # If the row is not zero, identify the pivot
        if np.all(np.abs(A[i, :]) < eps):
            continue

        for j in range(n):
            if abs(A[i, j]) >= eps:
                pivot = j
                break

        # Normalise the pivot
        A[i, :] /= A[i, pivot]

        # Clear entries above the pivot
        for j in range(i):
            factor = A[j, pivot]
            A[j, :] -= factor * A[i, :]

    # Set all small values to zero
    A[np.abs(A) < eps] = 0
    return A
