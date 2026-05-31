"""Рекурсивный (backtracking) поиск расстановки."""

from settings import BOARD_DIM, placement_grid
from grid_ops import diagonal_blocked


def backtrack_search(line: int = 0, occupied_files=None) -> bool:
    if occupied_files is None:
        occupied_files = set()

    if line == BOARD_DIM:
        return True

    for file_idx in range(BOARD_DIM):
        if file_idx in occupied_files:
            continue
        if diagonal_blocked(placement_grid, line, file_idx):
            continue

        placement_grid[line, file_idx] = 1
        occupied_files.add(file_idx)

        if backtrack_search(line + 1, occupied_files):
            return True

        placement_grid[line, file_idx] = 0
        occupied_files.remove(file_idx)

    return False
