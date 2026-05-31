"""Параметры доски и общее состояние сетки."""

BOARD_DIM = 8

# Глобальная матрица расстановки (используется двумя из трёх алгоритмов)
placement_grid = __import__("numpy").zeros((BOARD_DIM, BOARD_DIM), dtype=int)
