"""Случайный жадный перебор расстановок."""

import numpy as np

from settings import BOARD_DIM, placement_grid
from grid_ops import diagonal_blocked, every_row_has_piece


def stochastic_greedy() -> np.ndarray | None:
    row_order = np.arange(BOARD_DIM)
    col_order = np.arange(BOARD_DIM)
    attempt_limit = 20_000

    for _ in range(attempt_limit):
        placement_grid.fill(0)
        np.random.shuffle(row_order)
        np.random.shuffle(col_order)

        taken_rows: set[int] = set()
        taken_cols: set[int] = set()

        for row in row_order:
            for col in col_order:
                if row in taken_rows or col in taken_cols:
                    continue
                if diagonal_blocked(placement_grid, row, col):
                    continue

                placement_grid[row, col] = 1
                taken_rows.add(row)
                taken_cols.add(col)
                break

        if every_row_has_piece():
            return placement_grid.copy()

    return None
