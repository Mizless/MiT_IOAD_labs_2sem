"""Модель фигуры для кооперативного алгоритма."""

import random

from settings import BOARD_DIM


class ChessPiece:
    """Ферзь на фиксированной горизонтали, перемещается только по вертикали."""

    def __init__(self, line: int):
        self.line = line
        self.file = random.randint(0, BOARD_DIM - 1)

    def count_attacks(self, peers) -> int:
        hits = 0
        for peer in peers:
            if peer.line == self.line:
                continue
            if peer.file == self.file:
                hits += 1
            row_delta = abs(peer.line - self.line)
            col_delta = abs(peer.file - self.file)
            if row_delta == col_delta:
                hits += 1
        return hits

    def relocate_optimal(self, peers) -> None:
        """Переходит в столбец с минимальным числом атак (случайный при равенстве)."""
        lowest = BOARD_DIM
        candidates = []

        for file_idx in range(BOARD_DIM):
            self.file = file_idx
            attack_count = self.count_attacks(peers)

            if attack_count < lowest:
                lowest = attack_count
                candidates = [file_idx]
            elif attack_count == lowest:
                candidates.append(file_idx)

        self.file = random.choice(candidates)
