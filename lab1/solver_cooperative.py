"""Кооперативный алгоритм (агенты по очереди уменьшают число конфликтов)"""

import random

from settings import BOARD_DIM
from piece import ChessPiece
from grid_ops import agents_to_matrix


def cooperative_search(step_limit: int = 1000):
    roster = [ChessPiece(line=i) for i in range(BOARD_DIM)]

    for iteration in range(step_limit):
        per_piece = [unit.count_attacks(roster) for unit in roster]
        total = sum(per_piece)

        if total == 0:
            return agents_to_matrix(roster), iteration

        troubled = [unit for unit, cnt in zip(roster, per_piece) if cnt > 0]
        random.choice(troubled).relocate_optimal(roster)

    return None, step_limit
