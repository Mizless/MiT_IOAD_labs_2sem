"""Операции над шахматной сеткой."""

import numpy as np

from settings import BOARD_DIM, placement_grid


def diagonal_blocked(grid: np.ndarray, row_idx: int, col_idx: int) -> bool:
    """Проверяет, занята ли диагональ через клетку (row_idx, col_idx)."""
    offset_main = col_idx - row_idx
    offset_anti = (grid.shape[1] - col_idx - 1) - row_idx

    main_diag = np.diagonal(grid, offset=offset_main)
    anti_diag = np.diagonal(np.fliplr(grid), offset=offset_anti)

    return bool(np.any(main_diag == 1) or np.any(anti_diag == 1))


def every_row_has_piece() -> bool:
    """True, если в каждой строке стоит ровно одна фигура."""
    return bool(np.all(np.any(placement_grid, axis=1)))


def agents_to_matrix(pieces) -> np.ndarray:
    """Строит матрицу доски по списку фигур."""
    matrix = np.zeros((BOARD_DIM, BOARD_DIM), dtype=int)
    for piece in pieces:
        matrix[piece.line, piece.file] = 1
    return matrix


def reset_grid() -> None:
    placement_grid.fill(0)
