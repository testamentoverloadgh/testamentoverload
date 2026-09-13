import torch


def _as_float_tensor(matrix) -> torch.Tensor:
    """
    Return a floating-point 2-D tensor copy of matrix
    """
    tensor = torch.as_tensor(matrix)
    if tensor.dim() != 2:
        raise ValueError(f"expected a 2-D matrix, got {tensor.dim()} dimension(s)")
    if not torch.is_floating_point(tensor):
        tensor = tensor.to(torch.float64)
    return tensor.clone()


def _check_row(tensor: torch.Tensor, index: int, name: str) -> None:
    rows = tensor.shape[0]
    if not -rows <= index < rows:
        raise IndexError(f"{name}={index} is out of range for a matrix with {rows} rows")


def rowswap(matrix, source: int, target: int) -> torch.Tensor:
    """
    Swap rows in matrix
    """
    result = _as_float_tensor(matrix)
    _check_row(result, source, "source")
    _check_row(result, target, "target")
    if source == target:
        return result
    result[[source, target]] = result[[target, source]]
    return result

def rowscale(matrix, source: int, factor: float) -> torch.Tensor:
    """
    Scale row by a scalar
    """
    result = _as_float_tensor(matrix)
    _check_row(result, source, "source")
    if factor == 0:
        raise ValueError("scaling a row by 0 is not an elementary row operation")
    result[source] = result[source] * factor
    return result




def rowreplacement(matrix, row_i: int, row_j: int, j: float, k: float) -> torch.Tensor:
    """
    Replace row j with j * R_row_i + k * R_row_j

    Uses rowscale
    """
    result = _as_float_tensor(matrix)
    _check_row(result, row_i, "row_i")
    _check_row(result, row_j, "row_j")
    if row_i == row_j:
        raise ValueError("row_i and row_j must be different rows")

    scaled_i = rowscale(result, row_i, j)[row_i] if j != 0 else torch.zeros_like(result[row_i])
    scaled_j = rowscale(result, row_j, k)[row_j] if k != 0 else torch.zeros_like(result[row_j])
    result[row_j] = scaled_i + scaled_j
    return result



def rref(matrix, tol: float = 1e-9) -> torch.Tensor:
    """
    Returns the reduced row echelon form (RREF)
    
    The idea behind implementation is that first you find the pivot by picking the largest of the absolute values
    The second step is to use row swap to get it in the right spot.  Then use rowscale to get it reduced
    Then you use row replacement on the other rows
    """
    result = _as_float_tensor(matrix)
    rows, cols = result.shape
    pivot_row = 0

    for col in range(cols):
        if pivot_row >= rows:
            break

        candidates = result[pivot_row:, col].abs()
        offset = int(torch.argmax(candidates))
        if candidates[offset] <= tol:
            result[pivot_row:, col] = 0.0
            continue

        best_row = pivot_row + offset
        if best_row != pivot_row:
            result = rowswap(result, pivot_row, best_row)

        result = rowscale(result, pivot_row, 1.0 / result[pivot_row, col].item())
        result[pivot_row, col] = 1.0

        for row in range(rows):
            if row == pivot_row:
                continue
            entry = result[row, col].item()
            if abs(entry) > tol:
                result = rowreplacement(result, pivot_row, row, -entry, 1.0)
            result[row, col] = 0.0

        pivot_row += 1

    return result + 0.0


if __name__ == "__main__":
    torch.set_printoptions(precision=4, sci_mode=False)

    A = torch.tensor(
        [
            [1.0, 3.0, 0.0, 0.0, 3.0],
            [0.0, 0.0, 1.0, 0.0, 9.0],
            [0.0, 0.0, 0.0, 1.0, -4.0],
        ]
    )
    print("A =\n", A, "\n")

    step1 = rowswap(A, 0, 1)
    print("R1 <-> R2:\n", step1, "\n")

    step2 = rowscale(step1, 0, 1 / 3)
    print("(1/3) R1:\n", step2, "\n")

    step3 = rowreplacement(step2, 0, 2, -3.0, 1.0)
    print("R3 = -3 R1 + R3:\n", step3, "\n")

    print("A is unchanged:\n", A, "\n")
    print("rref(A):\n", rref(A), "\n")

    B = torch.tensor([[0.0, 2.0, 1.0, 4.0], [1.0, 1.0, 2.0, 6.0], [2.0, 2.0, 5.0, 13.0]])
    print("B =\n", B)
    print("rref(B):\n", rref(B))